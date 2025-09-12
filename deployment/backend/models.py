from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, func
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(320), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    role = Column(String(32), default="user", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class CoreMOF(Base):
    __tablename__ = "coremof"

    filename = Column(String, primary_key=True, index=True)
    LCD = Column(Float)
    PLD = Column(Float)
    LFPD = Column(Float)
    cm3_g = Column(Float)
    ASA_m2_cm3 = Column(Float)
    ASA_m2_g = Column(Float)
    NASA_m2_cm3 = Column(Float)
    NASA_m2_g = Column(Float)
    AV_VF = Column(Float)
    AV_cm3_g = Column(Float)
    NAV_cm3_g = Column(Float)
    Has_OMS = Column(Integer)
  

