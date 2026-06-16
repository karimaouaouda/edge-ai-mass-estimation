from crontab import CronTab

# Access the crontab of the current system user
cron = CronTab(user=True)

# Create a new job and point it to your Python script
job = cron.new(command='C:\\Users\\PC\\Desktop\\projects\\drovenai\\edge-ai-mass-estimation\\.venv\\Scripts\\python.exe C:\\Users\\PC\\Desktop\\projects\\drovenai\\src\\edge-ai-mass\\orchestration\\start.py')

# Set the schedule (e.g., every 15 minutes)
job.minute.every(1)

# Write the job to the system crontab
cron.write(user='PC')
print("Cron job created to run every 1 minute.")