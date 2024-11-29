import sqlite3

# 데이터베이스 연결
conn = sqlite3.connect('dataset/database.db')
cursor = conn.cursor()

# 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS apartments (
    id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT,
    price INTEGER,
    size FLOAT,
    rooms INTEGER,
    year_built INTEGER
)
''')

# 샘플 데이터 삽입
sample_data = [
    ('래미안아파트', '서울시 강남구', 1200000000, 84.5, 3, 2020),
    ('힐스테이트', '서울시 송파구', 980000000, 76.2, 2, 2019),
    ('자이아파트', '서울시 마포구', 850000000, 59.8, 2, 2021),
    ('푸르지오', '서울시 강서구', 720000000, 68.9, 3, 2018)
]

cursor.executemany('''
INSERT INTO apartments (name, location, price, size, rooms, year_built)
VALUES (?, ?, ?, ?, ?, ?)
''', sample_data)

# 변경사항 저장 및 연결 종료
conn.commit()
conn.close()