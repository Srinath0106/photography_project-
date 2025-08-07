import mysql.connector
from faker import Faker
import random
from datetime import timedelta
import re

fake = Faker()


cnx = mysql.connector.connect(
    user='root',
    password='srinath@25k',
    host='localhost',
    database='Photography_Project',  
)
cursor = cnx.cursor()


user_ids = []
photographer_ids = []
portfolio_ids = []
tag_ids = []

def insert_users(n=1000):
    print("Inserting users...")
    for _ in range(n):
        first_name = fake.first_name()
        last_name = fake.last_name()
        email = fake.unique.email()
        
        raw_phone = fake.phone_number()
       
        phone = re.sub(r'[^\d+]', '', raw_phone)[:20]
        
        registration_date = fake.date_between(start_date='-3y', end_date='today')
        last_login_date = registration_date + timedelta(days=random.randint(0, 1000))
        status = random.choice(['active', 'inactive'])

        cursor.execute('''
            INSERT INTO users (first_name, last_name, email, phone, registration_date, last_login_date, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (first_name, last_name, email, phone, registration_date, last_login_date, status))
        user_ids.append(cursor.lastrowid)

    cnx.commit()
    print(f"Inserted {n} users.")


def insert_photographers(n=1000):
    print("Inserting photographers...")
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    for _ in range(n):
        user_id = random.choice(user_ids)
        profile_name = fake.name()
        location = random.choice(locations)
        rating = round(random.uniform(2.5, 5.0), 2)
        status = random.choice(['active', 'inactive'])
        price_per_hour = round(random.uniform(50.00, 500.00), 2)
        created_at = fake.date_between(start_date='-3y', end_date='today')

        cursor.execute('''
            INSERT INTO photographers (user_id, profile_name, location, rating, status, price_per_hour, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (user_id, profile_name, location, rating, status, price_per_hour, created_at))
        photographer_ids.append(cursor.lastrowid)

    cnx.commit()
    print(f"Inserted {n} photographers.")


def insert_portfolios(n=1000):
    print("Inserting portfolios...")
    for _ in range(n):
        photographer_id = random.choice(photographer_ids)
        title = fake.sentence(nb_words=4)
        description = fake.text(max_nb_chars=200)
        created_at = fake.date_between(start_date='-3y', end_date='today')

        cursor.execute('''
            INSERT INTO portfolios (photographer_id, title, description, created_at)
            VALUES (%s, %s, %s, %s)
        ''', (photographer_id, title, description, created_at))
        portfolio_ids.append(cursor.lastrowid)

    cnx.commit()
    print(f"Inserted {n} portfolios.")


def insert_tags(n=1000):
    print("Inserting tags...")
    for _ in range(n):
        tag_name = fake.word()
        try:
            cursor.execute('''
                INSERT INTO tags (tag_name) VALUES (%s)
            ''', (tag_name,))
            tag_ids.append(cursor.lastrowid)
        except mysql.connector.errors.IntegrityError:
           
            continue

    cnx.commit()
    print(f"Inserted {len(tag_ids)} unique tags out of {n} attempts.")


def insert_portfolio_tags(n=1000):
    print("Inserting portfolio_tags...")
    for _ in range(n):
        portfolio_id = random.choice(portfolio_ids)
        tag_id = random.choice(tag_ids)

        cursor.execute('''
            INSERT IGNORE INTO portfolio_tags (portfolio_id, tag_id)
            VALUES (%s, %s)
        ''', (portfolio_id, tag_id))

    cnx.commit()
    print("Inserted portfolio_tags.")


def insert_bookings(n=1000):
    print("Inserting bookings...")
    categories = ['Wedding', 'Corporate', 'Birthday', 'Concert', 'Sports']

    for _ in range(n):
        user_id = random.choice(user_ids)
        photographer_id = random.choice(photographer_ids)
        booking_date = fake.date_between(start_date='-2y', end_date='today')
        event_category = random.choice(categories)
        total_amount = round(random.uniform(100, 5000), 2)
        status = random.choice(['confirmed', 'pending', 'cancelled'])

        cursor.execute('''
            INSERT INTO bookings (user_id, photographer_id, booking_date, event_category, total_amount, status)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (user_id, photographer_id, booking_date, event_category, total_amount, status))

    cnx.commit()
    print(f"Inserted {n} bookings.")


def insert_search_logs(n=1000):
    print("Inserting search_logs...")
    search_terms = ['wedding photographer', 'event photographer', 'corporate event', 'outdoor photography', 'portrait']

    for _ in range(n):
        user_id = random.choice(user_ids)
        search_term = random.choice(search_terms)
        search_date = fake.date_time_between(start_date='-2y', end_date='now')
        results_count = random.randint(0, 100)
        results_clicked = random.randint(0, min(results_count, 20))

        cursor.execute('''
            INSERT INTO search_logs (user_id, search_term, search_date, results_count, results_clicked)
            VALUES (%s, %s, %s, %s, %s)
        ''', (user_id, search_term, search_date, results_count, results_clicked))

    cnx.commit()
    print(f"Inserted {n} search logs.")


def insert_clicks(n=1000):
    print("Inserting clicks...")
    for _ in range(n):
        user_id = random.choice(user_ids)
        photographer_id = random.choice(photographer_ids)
        click_date = fake.date_time_between(start_date='-2y', end_date='now')

        cursor.execute('''
            INSERT INTO clicks (user_id, photographer_id, click_date)
            VALUES (%s, %s, %s)
        ''', (user_id, photographer_id, click_date))

    cnx.commit()
    print(f"Inserted {n} clicks.")


def insert_shortlists(n=1000):
    print("Inserting shortlists...")
    for _ in range(n):
        user_id = random.choice(user_ids)
        photographer_id = random.choice(photographer_ids)
        added_date = fake.date_between(start_date='-2y', end_date='today')

        cursor.execute('''
            INSERT INTO shortlists (user_id, photographer_id, added_date)
            VALUES (%s, %s, %s)
        ''', (user_id, photographer_id, added_date))

    cnx.commit()
    print(f"Inserted {n} shortlists.")


def insert_pricing_history(n=1000):
    print("Inserting pricing_history...")
    for _ in range(n):
        photographer_id = random.choice(photographer_ids)
        price = round(random.uniform(50, 1000), 2)
        effective_date = fake.date_between(start_date='-3y', end_date='today')

        cursor.execute('''
            INSERT INTO pricing_history (photographer_id, price, effective_date)
            VALUES (%s, %s, %s)
        ''', (photographer_id, price, effective_date))

    cnx.commit()
    print(f"Inserted {n} pricing history records.")


# Run insertion procedures in order respecting foreign keys
insert_users()
insert_photographers()
insert_portfolios()
insert_tags()
insert_portfolio_tags()
insert_bookings()
insert_search_logs()
insert_clicks()
insert_shortlists()
insert_pricing_history()

cursor.close()
cnx.close()
print("All data inserted successfully.")
