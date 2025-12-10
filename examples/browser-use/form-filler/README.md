# Form Filler Examples

Browser automation that fills out web forms and learns efficient strategies over time.

## Files

- **ace_form_filler.py** - Form filling WITH ACE learning
- **baseline_form_filler.py** - Form filling WITHOUT ACE
- **ace_browser_use.py** - Advanced ACE browser agent with HTTP server
- **baseline_browser_use.py** - Basic browser agent
- **form.html** - Example HTML form for testing
- **task1_flight_search.txt** - Flight search task definition
- **task2_form.txt** - Form filling task definition

## Quick Run

```bash
# Advanced ACE agent with form.html
uv run python examples/browser-use/form-filler/ace_browser_use.py

# With custom task file
uv run python examples/browser-use/form-filler/ace_browser_use.py --task-file task1_flight_search.txt

# Basic baseline (no learning)
uv run python examples/browser-use/form-filler/baseline_browser_use.py
```

## How It Works

### ace_browser_use.py Features

1. **Local HTTP Server**: Automatically serves form.html on localhost:8765
2. **Strategy Planning**: Agent creates step-by-step form filling plan
3. **Browser Execution**: Browser-use agent follows the plan
4. **Learning Loop**: Reflects on performance and updates skillbook
5. **Token Tracking**: Monitors ACE and browser-use token usage

### Task Files

**task1_flight_search.txt**: Search for flights on Google Flights
```
Go to https://www.google.com/travel/flights
Search for cheapest flights from Zürich to New York
Leave: November 1st, Return: November 8th
```

**task2_form.txt**: Fill out local form
```
Go to http://127.0.0.1:8765/form.html
Sign up with:
- First Name: Miguel
- Last Name: Martinez
- Email: miguel.martinez@example.com
Submit the form
```

## Configuration

### Model Selection

```python
# ACE roles (planning, reflection, learning)
llm = LiteLLMClient(model="gpt-4o-mini", temperature=0.7)

# Browser-use agent (execution)
environment = BrowserUseEnvironment(
    headless=False,
    model="gpt-4o-mini"
)
```

### HTTP Server

The form.html is served automatically:
- Starts on port 8765
- Serves from script's directory
- Accessible at http://127.0.0.1:8765/form.html

## Expected Results

**First task:**
```
Agent creates action plan (dict format):
{
  "1": "Navigate to the form URL",
  "2": "Fill in the First Name field",
  "3": "Fill in the Last Name field",
  "4": "Fill in the Email field",
  "5": "Click the Submit button"
}
```

**Browser executes plan:**
- Takes 5-8 steps typically
- May encounter errors on first try
- Provides detailed feedback

**After learning:**
- More efficient click patterns
- Better error handling
- Faster form completion

## Features

### Token Usage Tracking

Automatically tracks and reports:
- **ACE tokens**: Agent + Reflector + SkillManager
- **Browser-use tokens**: Browser automation agent
- **Per-task breakdown**: See costs for each sample

```
💰 Token Usage:
   🧠 ACE Tokens:         2,450 total     490.0 per task
   🤖 Browser Tokens:     12,834 total   2,566.8 per task

   Role breakdown:
   🎯 Agent:          850 tokens  (strategy planning)
   🔍 Reflector:      720 tokens  (performance analysis)
   📝 SkillManager:   880 tokens  (skillbook updates)
```

### Opik Integration

If Opik is installed:
- Automatic trace logging
- Cost monitoring
- Performance analytics
- View at https://www.comet.com/opik

## Customization

### Create Your Own Form

1. Edit `form.html` or create new HTML file
2. Update `task2_form.txt` with form details
3. Run: `python ace_browser_use.py --task-file your_task.txt`

### Add More Tasks

```python
# In the main() function
samples = []
for i in range(10):  # Increase number of iterations
    samples.append(Sample(
        question=question,
        ground_truth="SUCCESS",
        context=task_content
    ))
```

## Output Format

The Generator expects to return a dictionary:
```python
{
    "1": "Step 1 description",
    "2": "Step 2 description",
    "3": "Step 3 description"
}
```

This gets converted to a browser-use prompt:
```
Task: [original task]

Follow these steps:
1: Step 1 description
2: Step 2 description
3: Step 3 description
```

## Troubleshooting

**HTTP server fails to start**
- Port 8765 may be in use
- Change port in `_start_http_server(port=8765)`

**Form not loading**
- Check browser console for errors
- Verify form.html is in same directory
- Try absolute path to form.html

**JSON parsing errors**
- Agent should return dict format
- Check `agent_output.final_answer` structure
- Enable debug output with `print("AGENT OUTPUT:", agent_output)`

## Advanced Usage

### Structured Output

The script extracts various outputs from browser history:
- `model_outputs()`: Agent's thoughts and goals
- `final_result()`: Task completion status
- `is_successful()`: Boolean success flag
- `number_of_steps()`: Step count

### Debug Mode

Uncomment in the code:
```python
# print_history_details(result)  # Line 134
```

This shows comprehensive execution details:
- All actions taken
- URLs visited
- Errors encountered
- Model reasoning
- Timing information

## Next Steps

1. Try both task files
2. Create your own form and task
3. Compare ACE vs baseline performance
4. Monitor token usage with Opik
5. Build your own form-filling use case!
