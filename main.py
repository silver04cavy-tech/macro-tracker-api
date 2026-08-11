from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlmodel import Field, SQLModel, Session, create_engine, select
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
import os

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_4pMkoHfaeSt8@ep-patient-mouse-aypszbto.c-5.us-east-2.aws.neon.tech/neondb?sslmode=require"
)

engine = create_engine(DATABASE_URL, echo=True)

class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    hashed_password: str

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

def get_session():
    with Session(engine) as session:
        yield session

# --- SECURITY & AUTH CONFIG ---
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

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

# --- FASTAPI APP SETUP ---
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

# --- MEAL LOGGING ROUTES ---
@app.post("/log-meal")
async def log_meal(
    description: str = Form(""),
    file: UploadFile | None = File(None),
    current_user: str = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    meal_desc = description if description else "Uploaded Meal Photo"
    cals, protein, carbs, fats = 450, 35, 40, 15

    new_meal = Meal(
        username=current_user,
        description=meal_desc,
        calories=cals,
        protein=protein,
        carbs=carbs,
        fats=fats
    )
    session.add(new_meal)
    session.commit()
    session.refresh(new_meal)

    return {"message": "Meal logged successfully", "meal": new_meal}

@app.get("/meals")
def get_meals(current_user: str = Depends(get_current_user), session: Session = Depends(get_session)):
    meals = session.exec(select(Meal).where(Meal.username == current_user)).all()
    return meals