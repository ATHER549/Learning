How to Run the Agent
Follow these steps to get your event management agent running.

Step 1: Install Dependencies
Open your terminal and run the following command in your project directory:
pip install -r requirements.txt

Step 2: Set Your API Key
Make sure your .env file is in the same directory and contains your valid OpenAI API key.

Step 3: Run the Script
Execute the Python script from your terminal:
python event_agent.py

Step 4: Interact with the Agent
The script will prompt you for input. You can now make event scheduling requests.

Example Interactions
Here are some examples of how you can interact with the agent and the expected output.

Example:
You: Create an event: Presentation on new features next Friday at 2 PM, in the main hall, with the sales team.
Assistant: The event "Presentation on new features" has been scheduled for Friday, September 26, 2025, at 2:00 PM in the main hall with the sales team. It has been added to your calendar
