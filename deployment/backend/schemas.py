from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    role: str
    is_active: bool
    class Config:
        from_attributes = True  # Pydantic v2

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CoreMOFSchema(BaseModel):
    filename: str
    LCD: float | None = None
    PLD: float | None = None
    LFPD: float | None = None
    cm3_g: float | None = None
    ASA_m2_cm3: float | None = None
    ASA_m2_g: float | None = None
    NASA_m2_cm3: float | None = None
    NASA_m2_g: float | None = None
    AV_VF: float | None = None
    AV_cm3_g: float | None = None
    NAV_cm3_g: float | None = None
    Has_OMS: int | None = None
    # class Config:
    #     orm_mode = True
    class ConfigDict:
        from_attributes=True




