#!/bin/bash

# Configuration
DB_PATH="/var/www/sol-app/backend/data/sol.db"
BACKUP_DIR="/var/backups/sol_db"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/sol_backup_${TIMESTAMP}.sqlite"
LOG_FILE="/var/log/sol_backup.log"

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"

echo "=== Backup started at $(date) ===" >> "$LOG_FILE"

# 1. Verify source SQLite database exists
if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: SQLite database file not found at ${DB_PATH}" >> "$LOG_FILE"
    exit 1
fi

# 2. Run Safe online backup using local sqlite3 binary
# This locks transactions gracefully, creating a clean snapshot
sqlite3 "$DB_PATH" ".backup '${BACKUP_FILE}'" >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    # 3. Compress the backup to save disk space
    gzip "$BACKUP_FILE"
    echo "SUCCESS: Database backup completed successfully: ${BACKUP_FILE}.gz" >> "$LOG_FILE"
else
    echo "ERROR: sqlite3 backup process failed." >> "$LOG_FILE"
    exit 1
fi

# 4. Clean up backups older than 14 days to prevent storage leak
find "$BACKUP_DIR" -name "sol_backup_*.sqlite.gz" -type f -mtime +14 -delete
echo "SUCCESS: Rotated and removed old backups." >> "$LOG_FILE"

echo "=== Backup completed at $(date) ===" >> "$LOG_FILE"
