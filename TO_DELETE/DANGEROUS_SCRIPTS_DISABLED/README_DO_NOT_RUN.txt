⚠️ CRITICAL WARNING ⚠️

These scripts DELETE all NFL/NBA/NHL schedules from the database.

DO NOT RUN ANY OF THESE SCRIPTS.

The schedules in sports_predictions_original.db are FIXED and should NEVER be modified.

Running any script in this directory will:
- DELETE all existing games for that sport
- Cause data loss
- Break the production system
- Cost money to fix

If you need to update schedules:
1. Make changes in a STAGING database first
2. Get explicit approval
3. Manually backup sports_predictions_original.db
4. Test thoroughly before touching production

Date disabled: October 31, 2025
Reason: Repeated accidental schedule deletions
