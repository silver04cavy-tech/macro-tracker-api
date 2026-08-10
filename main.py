import os
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import jwt
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from google import genai
from google.genai import types
from passlib.context import CryptContext
from PIL import Image
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Field as SQLField, Session, create_engine, select

# --- CONFIGURATION & SECURITY ---
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-development-key-change-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///meals.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

class User(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    username: str = SQLField(index=True, unique=True)
    hashed_password: str

class MealLog(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(foreign_key="user.id")
    timestamp: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))
    total_calories: int
    total_protein_g: float
    total_carbs_g: float
    total_fats_g: float
    raw_json_analysis: str

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

# --- AUTH HELPERS ---
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = session.exec(select(User).where(User.username == username)).first()
    if user is None:
        raise credentials_exception
    return user

# --- PYDANTIC SCHEMAS ---
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class FoodItem(BaseModel):
    name: str = Field(description="Name of the food item detected")
    estimated_weight_g: float = Field(description="Estimated weight or portion size in grams")
    calories: int = Field(description="Total calories in kcal")
    protein_g: float = Field(description="Protein content in grams")
    carbs_g: float = Field(description="Carbohydrate content in grams")
    fats_g: float = Field(description="Fat content in grams")
    confidence: str = Field(description="Confidence rating: low, medium, or high")

class MealAnalysisResponse(BaseModel):
    foods: List[FoodItem]
    total_calories: int
    total_protein_g: float
    total_carbs_g: float
    total_fats_g: float
    additional_notes: Optional[str] = Field(
        description="Follow-up suggestions, such as checking for added oils, butter, or dressings."
    )

class MealManualCreate(BaseModel):
    meal_name: str
    calories: int
    protein_g: float
    carbs_g: float
    fats_g: float

# --- FASTAPI APP ---
app = FastAPI(title="Macro Tracker Vision API")

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

# --- AUTH ENDPOINTS ---
@app.post("/register", response_model=Token)
def register(user_data: UserCreate, session: Session = Depends(get_session)):
    try:
        existing_user = session.exec(select(User).where(User.username == user_data.username)).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already registered")

        user = User(
            username=user_data.username,
            hashed_password=get_password_hash(user_data.password)
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        access_token = create_access_token(data={"sub": user.username})
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.username == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# --- MEAL ENDPOINTS ---
def analyze_image_sync(api_key: str, image: Image.Image) -> str:
    client = genai.Client(api_key=api_key)
    
    prompt = (
        "You are a professional registered dietitian. Analyze the provided meal photo. "
        "Identify each food item, estimate its portion size in grams, and calculate its "
        "macronutrients (calories, protein, carbs, fats). Sum the totals at the end."
    )
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MealAnalysisResponse,
        ),
    )
    return response.text

@app.post("/analyze-meal", response_model=MealAnalysisResponse)
async def analyze_meal(
    file: UploadFile = File(...), 
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is missing.")

    try:
        image = Image.open(file.file)
        json_output = await asyncio.to_thread(analyze_image_sync, api_key, image)
        parsed_response = MealAnalysisResponse.model_validate_json(json_output)

        db_meal = MealLog(
            user_id=current_user.id,
            total_calories=parsed_response.total_calories,
            total_protein_g=parsed_response.total_protein_g,
            total_carbs_g=parsed_response.total_carbs_g,
            total_fats_g=parsed_response.total_fats_g,
            raw_json_analysis=json_output
        )
        session.add(db_meal)
        session.commit()

        return parsed_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log-meal-manual", response_model=MealLog)
def log_meal_manual(
    meal: MealManualCreate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    raw_info = f'{{"manual_entry": true, "meal_name": "{meal.meal_name}"}}'
    db_meal = MealLog(
        user_id=current_user.id,
        total_calories=meal.calories,
        total_protein_g=meal.protein_g,
        total_carbs_g=meal.carbs_g,
        total_fats_g=meal.fats_g,
        raw_json_analysis=raw_info
    )
    session.add(db_meal)
    session.commit()
    session.refresh(db_meal)
    return db_meal

@app.get("/meals", response_model=List[MealLog])
def read_meals(
    current_user: User = Depends(get_current_user), 
    session: Session = Depends(get_session)
):
    meals = session.exec(select(MealLog).where(MealLog.user_id == current_user.id)).all()
    return meals