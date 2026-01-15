-- DuckDB schema for payments analytics mart.

CREATE TABLE IF NOT EXISTS merchants (
    merchant_id BIGINT PRIMARY KEY,
    industry VARCHAR,
    region VARCHAR,
    merchant_size VARCHAR,
    sales_channel VARCHAR,
    signup_date DATE
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id BIGINT PRIMARY KEY,
    segment VARCHAR,
    region VARCHAR,
    join_date DATE
);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id BIGINT PRIMARY KEY,
    ts TIMESTAMP,
    txn_date DATE,
    merchant_id BIGINT REFERENCES merchants(merchant_id),
    customer_id BIGINT REFERENCES customers(customer_id),
    amount DOUBLE,
    currency VARCHAR,
    approved BOOLEAN,
    payment_method VARCHAR,
    decline_reason VARCHAR,
    fee DOUBLE,
    channel VARCHAR,
    partition_year INTEGER,
    partition_month INTEGER
);

CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id BIGINT PRIMARY KEY,
    campaign_name VARCHAR,
    channel VARCHAR,
    start_date DATE,
    end_date DATE,
    budget DOUBLE
);
