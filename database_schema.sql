CREATE DATABASE IF NOT EXISTS neuroguard;
USE neuroguard;
CREATE TABLE IF NOT EXISTS fatigue_sessions (
 id INT AUTO_INCREMENT PRIMARY KEY,
 session_time DATETIME NOT NULL,
 fatigue_level VARCHAR(40) NOT NULL,
 confidence DECIMAL(7,3) NOT NULL,
 fatigue_score INT NOT NULL,
 risk_score INT NOT NULL,
 drowsiness_score DECIMAL(7,3) NOT NULL,
 alertness_score DECIMAL(7,3) NOT NULL,
 delta_power DECIMAL(12,8),
 theta_power DECIMAL(12,8),
 alpha_power DECIMAL(12,8),
 beta_power DECIMAL(12,8),
 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_session_time ON fatigue_sessions(session_time);
