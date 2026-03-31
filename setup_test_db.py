"""
setup_test_db.py
================
Script de création et remplissage de la base de données de test
pour le projet AI-Assisted SQL Query Optimization.

Génère :
  - 4 tables : customers, products, orders, order_items
  - ~50 000 lignes au total pour tester les performances

Usage :
    python setup_test_db.py
"""

import psycopg2
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

# ── Connexion ──────────────────────────────────────────────────
conn = psycopg2.connect(
    host     = os.getenv("DB_HOST", "localhost"),
    port     = os.getenv("DB_PORT", "5432"),
    dbname   = os.getenv("DB_NAME", "postgres"),
    user     = os.getenv("DB_USER", "postgres"),
    password = os.getenv("DB_PASSWORD", "")
)
conn.autocommit = True
cur = conn.cursor()

print("✅ Connecté à PostgreSQL")
print("🗑️  Suppression des anciennes tables...")

# ── Suppression des anciennes tables ──────────────────────────
cur.execute("""
    DROP TABLE IF EXISTS order_items CASCADE;
    DROP TABLE IF EXISTS orders CASCADE;
    DROP TABLE IF EXISTS products CASCADE;
    DROP TABLE IF EXISTS customers CASCADE;
""")

# ── Création des tables ───────────────────────────────────────
print("📋 Création des tables...")

cur.execute("""
CREATE TABLE customers (
    id         SERIAL PRIMARY KEY,
    name       VARCHAR(100) NOT NULL,
    email      VARCHAR(150) UNIQUE NOT NULL,
    region     VARCHAR(50)  NOT NULL,
    country    VARCHAR(50)  NOT NULL,
    created_at TIMESTAMP    NOT NULL DEFAULT NOW()
);
""")

cur.execute("""
CREATE TABLE products (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(150) NOT NULL,
    category    VARCHAR(80)  NOT NULL,
    price       NUMERIC(10,2) NOT NULL,
    stock       INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);
""")

cur.execute("""
CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    status      VARCHAR(30) NOT NULL DEFAULT 'pending',
    order_date  DATE NOT NULL,
    total       NUMERIC(12,2) NOT NULL DEFAULT 0,
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);
""")

cur.execute("""
CREATE TABLE order_items (
    id         SERIAL PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL DEFAULT 1,
    unit_price NUMERIC(10,2) NOT NULL
);
""")

print("✅ Tables créées")

# ── Données de référence ──────────────────────────────────────
REGIONS   = ['EU', 'NA', 'ASIA', 'LATAM', 'MENA']
COUNTRIES = {
    'EU':    ['France', 'Germany', 'Spain', 'Italy', 'Netherlands'],
    'NA':    ['USA', 'Canada', 'Mexico'],
    'ASIA':  ['Japan', 'China', 'India', 'South Korea'],
    'LATAM': ['Brazil', 'Argentina', 'Colombia'],
    'MENA':  ['Morocco', 'UAE', 'Saudi Arabia', 'Egypt'],
}
CATEGORIES = ['Electronics', 'Clothing', 'Books', 'Food', 'Sports', 'Home', 'Toys']
STATUSES   = ['pending', 'confirmed', 'shipped', 'delivered', 'cancelled']

# ── Insertion des clients (10 000) ────────────────────────────
print("👤 Insertion de 10 000 clients...")
customers = []
for i in range(1, 10001):
    region  = random.choice(REGIONS)
    country = random.choice(COUNTRIES[region])
    customers.append((
        f"Customer {i}",
        f"customer{i}@example.com",
        region,
        country,
        datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1500))
    ))

cur.executemany("""
    INSERT INTO customers (name, email, region, country, created_at)
    VALUES (%s, %s, %s, %s, %s)
""", customers)
print(f"  ✅ 10 000 clients insérés")

# ── Insertion des produits (500) ──────────────────────────────
print("📦 Insertion de 500 produits...")
products = []
for i in range(1, 501):
    category = random.choice(CATEGORIES)
    products.append((
        f"{category} Product {i}",
        category,
        round(random.uniform(5.0, 999.99), 2),
        random.randint(0, 500),
        datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1000))
    ))

cur.executemany("""
    INSERT INTO products (name, category, price, stock, created_at)
    VALUES (%s, %s, %s, %s, %s)
""", products)
print(f"  ✅ 500 produits insérés")

# ── Insertion des commandes (30 000) ──────────────────────────
print("🛒 Insertion de 30 000 commandes...")
orders = []
for i in range(1, 30001):
    orders.append((
        random.randint(1, 10000),
        random.choice(STATUSES),
        (datetime(2022, 1, 1) + timedelta(days=random.randint(0, 1100))).date(),
        round(random.uniform(10.0, 5000.0), 2),
        datetime(2022, 1, 1) + timedelta(days=random.randint(0, 1100))
    ))

cur.executemany("""
    INSERT INTO orders (customer_id, status, order_date, total, created_at)
    VALUES (%s, %s, %s, %s, %s)
""", orders)
print(f"  ✅ 30 000 commandes insérées")

# ── Insertion des lignes de commande (100 000) ────────────────
print("📋 Insertion de 100 000 order_items (patient...)...")
items = []
for order_id in range(1, 30001):
    n_items = random.randint(1, 5)
    for _ in range(n_items):
        product_id = random.randint(1, 500)
        quantity   = random.randint(1, 10)
        unit_price = round(random.uniform(5.0, 999.99), 2)
        items.append((order_id, product_id, quantity, unit_price))
    if len(items) >= 100000:
        break

cur.executemany("""
    INSERT INTO order_items (order_id, product_id, quantity, unit_price)
    VALUES (%s, %s, %s, %s)
""", items)
print(f"  ✅ {len(items):,} order_items insérés")

# ── Statistiques finales ──────────────────────────────────────
print("\n📊 Récapitulatif des données insérées :")
for table in ['customers', 'products', 'orders', 'order_items']:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    print(f"   {table:20s} : {count:>10,} lignes")

print("\n✅ Base de données de test prête !")
print("\n💡 Exemples de requêtes SQL à tester dans ton app :")
print("""
-- Test 1 : Requête lente (sans index)
SELECT * FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.order_date >= '2023-01-01'
  AND c.region = 'EU';

-- Test 2 : Filtre sur une catégorie
SELECT p.name, p.price, oi.quantity
FROM order_items oi
JOIN products p ON oi.product_id = p.id
WHERE p.category = 'Electronics'
  AND oi.quantity > 5;

-- Test 3 : Multi-jointure complexe
SELECT c.name, c.region, o.status, o.total
FROM customers c
JOIN orders o ON c.id = o.customer_id
JOIN order_items oi ON o.id = oi.order_id
WHERE c.country = 'Morocco'
  AND o.status = 'delivered'
  AND o.order_date BETWEEN '2023-01-01' AND '2024-01-01';
""")

cur.close()
conn.close()