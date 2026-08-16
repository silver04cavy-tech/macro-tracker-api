from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlmodel import Field, SQLModel, Session, create_engine, select
from sqlalchemy import text
from passlib.context import CryptContext
from google import genai
from google.genai import types
from PIL import Image
import jwt
from datetime import datetime, timedelta, date
import os
import io
import json
import urllib.request

# --- ENVIRONMENT CONFIG ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY environment variable is not set.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

engine = create_engine(DATABASE_URL, echo=False)

# --- MODELS ---
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    hashed_password: str
    target_calories: int = Field(default=2000)
    target_protein: int = Field(default=150)
    target_carbs: int = Field(default=200)
    target_fats: int = Field(default=65)

class Meal(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True)
    description: str
    calories: int
    protein: int
    carbs: int
    fats: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    
    # Auto-add missing columns to existing user table
    columns = [
        ("target_calories", "INTEGER DEFAULT 2000"),
        ("target_protein", "INTEGER DEFAULT 150"),
        ("target_carbs", "INTEGER DEFAULT 200"),
        ("target_fats", "INTEGER DEFAULT 65")
    ]
    
    with engine.connect() as conn:
        for col_name, col_type in columns:
            try:
                conn.execute(text(f'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS {col_name} {col_type};'))
                conn.commit()
            except Exception as e:
                print(f"Migration notice for {col_name}: {e}")

def get_session():
    with Session(engine) as session:
        yield session

# --- AUTH CONFIG ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- GEMINI CLIENT INIT ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# --- FASTAPI APP ---
app = FastAPI(title="Macro Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/")
def read_root():
    return {"message": "Macro Tracker API is live and running!"}

# --- AUTH ROUTES ---
@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(user_data: dict, session: Session = Depends(get_session)):
    username = user_data.get("username")
    password = user_data.get("password")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    existing_user = session.exec(select(User).where(User.username == username)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed_pw = hash_password(password)
    new_user = User(username=username, hashed_password=hashed_pw)
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return {"message": "User created successfully", "user_id": new_user.id}

@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.username == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# --- USER GOALS ROUTES ---
@app.get("/user/goals")
def get_user_goals(current_user: str = Depends(get_current_user), session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.username == current_user)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "calories": user.target_calories,
        "protein": user.target_protein,
        "carbs": user.target_carbs,
        "fats": user.target_fats
    }

@app.put("/user/goals")
def update_user_goals(goals_data: dict, current_user: str = Depends(get_current_user), session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.username == current_user)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.target_calories = int(goals_data.get("calories", user.target_calories))
    user.target_protein = int(goals_data.get("protein", user.target_protein))
    user.target_carbs = int(goals_data.get("carbs", user.target_carbs))
    user.target_fats = int(goals_data.get("fats", user.target_fats))

    session.add(user)
    session.commit()
    session.refresh(user)
    return {"message": "Goals updated successfully"}

# --- MEAL ROUTES ---
@app.post("/log-meal")
async def log_meal(
    description: str = Form(""),
    file: UploadFile | None = File(None),
    current_user: str = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    if not description and not file:
        raise HTTPException(status_code=400, detail="Please provide a photo or text description.")

    meal_description = description if description else "Logged Meal"
    cals, protein, carbs, fats = 300, 20, 30, 10

    if ai_client:
        try:
            prompt = """
            Analyze this meal from the provided text description and/or image.
            Estimate the total nutritional values for the entire portion.
            Return ONLY a valid JSON object with these exact keys:
            {
                "description": "Short name/summary of the food items",
                "calories": integer,
                "protein": integer (grams),
                "carbs": integer (grams),
                "fats": integer (grams)
            }
            Do not include any markdown formatting, backticks, or extra text.
            """

            contents = [prompt]
            if description:
                contents.append(f"User description: {description}")
            if file:
                image_bytes = await file.read()
                pil_image = Image.open(io.BytesIO(image_bytes))
                contents.append(pil_image)

            response = ai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )

            ai_data = json.loads(response.text)
            meal_description = ai_data.get("description", meal_description)
            cals = int(ai_data.get("calories", cals))
            protein = int(ai_data.get("protein", protein))
            carbs = int(ai_data.get("carbs", carbs))
            fats = int(ai_data.get("fats", fats))

        except Exception as e:
            print(f"Gemini API Error: {e}")
            if description:
                meal_description = description

    new_meal = Meal(
        username=current_user,
        description=meal_description,
        calories=cals,
        protein=protein,
        carbs=carbs,
        fats=fats
    )

    session.add(new_meal)
    session.commit()
    session.refresh(new_meal)
    return {"message": "Meal logged successfully", "meal": new_meal}

@app.post("/log-meal-manual")
def log_meal_manual(
    meal_data: dict,
    current_user: str = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    desc = meal_data.get("description", "Manual Meal")
    cals = int(meal_data.get("calories", 0))
    prot = int(meal_data.get("protein", 0))
    carbs = int(meal_data.get("carbs", 0))
    fats = int(meal_data.get("fats", 0))

    new_meal = Meal(
        username=current_user,
        description=desc,
        calories=cals,
        protein=prot,
        carbs=carbs,
        fats=fats
    )

    session.add(new_meal)
    session.commit()
    session.refresh(new_meal)
    return {"message": "Manual meal logged", "meal": new_meal}

@app.get("/barcode/{code}")
def get_barcode_data(code: str, current_user: str = Depends(get_current_user)):
    url = f"https://world.openfoodfacts.org/api/v2/product/{code}.json"
    req = urllib.request.Request(
        url, 
        headers={"User-Agent": "MacroTrackerApp/1.0 (test@macrotracker.local)"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
        if data.get("status") != 1:
            raise HTTPException(status_code=404, detail="Product not found in database.")
            
        product = data.get("product", {})
        nutriments = product.get("nutriments", {})
        name = product.get("product_name", f"Barcode Item ({code})")
        
        calories = int(round(float(nutriments.get("energy-kcal_serving") or nutriments.get("energy-kcal_100g") or 0)))
        protein = int(round(float(nutriments.get("proteins_serving") or nutriments.get("proteins_100g") or 0)))
        carbs = int(round(float(nutriments.get("carbohydrates_serving") or nutriments.get("carbohydrates_100g") or 0)))
        fats = int(round(float(nutriments.get("fat_serving") or nutriments.get("fat_100g") or 0)))
        
        return {
            "description": name,
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "fats": fats
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch product: {str(e)}")

@app.get("/meals")
def get_meals(
    selected_date: str | None = Query(None),
    current_user: str = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    query = select(Meal).where(Meal.username == current_user)
    
    if selected_date:
        try:
            target_dt = date.fromisoformat(selected_date)
            start_dt = datetime.combine(target_dt, datetime.min.time())
            end_dt = datetime.combine(target_dt, datetime.max.time())
            query = query.where(Meal.timestamp >= start_dt).where(Meal.timestamp <= end_dt)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    meals = session.exec(query.order_by(Meal.timestamp.desc())).all()
    return meals

@app.get("/meals/export")
def export_meals(current_user: str = Depends(get_current_user), session: Session = Depends(get_session)):
    meals = session.exec(select(Meal).where(Meal.username == current_user).order_by(Meal.timestamp.asc())).all()
    return meals

@app.put("/meals/{meal_id}")
def update_meal(
    meal_id: int,
    meal_data: dict,
    current_user: str = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    meal = session.get(Meal, meal_id)
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")
    if meal.username != current_user:
        raise HTTPException(status_code=403, detail="Not authorized")

    meal.description = meal_data.get("description", meal.description)
    meal.calories = int(meal_data.get("calories", meal.calories))
    meal.protein = int(meal_data.get("protein", meal.protein))
    meal.carbs = int(meal_data.get("carbs", meal.carbs))
    meal.fats = int(meal_data.get("fats", meal.fats))

    session.add(meal)
    session.commit()
    session.refresh(meal)
    return {"message": "Meal updated", "meal": meal}

@app.delete("/meals/{meal_id}")
def delete_meal(
    meal_id: int,
    current_user: str = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    meal = session.get(Meal, meal_id)
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")
    if meal.username != current_user:
        raise HTTPException(status_code=403, detail="Not authorized")

    session.delete(meal)
    session.commit()
    return {"message": "Meal deleted successfully"}