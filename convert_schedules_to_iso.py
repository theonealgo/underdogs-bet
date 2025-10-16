#!/usr/bin/env python3
"""
Convert schedules.py dates from DD/MM/YYYY HH:MM to ISO YYYY-MM-DD HH:MM
This eliminates ambiguity and prevents month/day swapping issues.
"""

import re
from datetime import datetime

def convert_date(match):
    """Convert DD/MM/YYYY HH:MM to YYYY-MM-DD HH:MM"""
    date_str = match.group(1)
    try:
        dt = datetime.strptime(date_str, '%d/%m/%Y %H:%M')
        iso_format = dt.strftime('%Y-%m-%d %H:%M')
        return f"'date': '{iso_format}'"
    except:
        return match.group(0)  # Return original if parsing fails

# Read schedules.py
with open('models/schedules.py', 'r') as f:
    content = f.read()

# Convert all date entries
# Pattern: 'date': 'DD/MM/YYYY HH:MM'
pattern = r"'date':\s*'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})'"
converted_content = re.sub(pattern, convert_date, content)

# Update documentation to reflect new format
old_doc = "- date: Match date (YYYY-MM-DD format)"
new_doc = "- date: Match date (YYYY-MM-DD HH:MM format - ISO standard)"
converted_content = converted_content.replace(old_doc, new_doc)

# Write back
with open('models/schedules.py', 'w') as f:
    f.write(converted_content)

print("✅ Converted all dates to ISO format (YYYY-MM-DD HH:MM)")
print("✅ Updated documentation to reflect new format")
print("\nNext steps:")
print("1. Update parse_date() in import_schedules.py to handle YYYY-MM-DD HH:MM")
print("2. Re-run import_schedules.py")
