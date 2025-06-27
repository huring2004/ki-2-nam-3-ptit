CREATE DATABASE sql_injection_demo;
USE sql_injection_demo;
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50),
    password VARCHAR(50)
);
INSERT INTO users (username, password) VALUES ('admin', 'admin123');