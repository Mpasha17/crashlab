-- ShopCo analytics sample (fictional). Amounts are in US dollars.
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT NOT NULL);
CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL REFERENCES customers(id),
  order_date TEXT NOT NULL,          -- ISO date
  status TEXT NOT NULL,              -- paid | cancelled | refunded
  total REAL NOT NULL                -- order total in USD
);
INSERT INTO customers VALUES
 (1,'Ana Souza','Brazil'),(2,'Ben Fischer','Germany'),(3,'Chloe Martin','France'),
 (4,'Dev Patel','India'),(5,'Emma Weber','Germany'),(6,'Farid Haddad','France');
INSERT INTO orders VALUES
 (101,1,'2026-01-14','paid',120.00),(102,2,'2026-01-20','paid',89.50),
 (103,3,'2026-02-03','cancelled',300.00),(104,4,'2026-02-11','paid',45.25),
 (105,5,'2026-02-27','paid',210.00),(106,2,'2026-03-02','paid',64.00),
 (107,6,'2026-03-09','refunded',150.00),(108,1,'2026-03-15','paid',99.99),
 (109,3,'2026-03-21','paid',35.00),(110,5,'2026-03-28','cancelled',500.00),
 (111,4,'2026-04-04','paid',75.50),(112,6,'2026-04-18','paid',180.00),
 (113,2,'2026-04-25','refunded',40.00),(114,5,'2026-05-06','paid',130.00);
