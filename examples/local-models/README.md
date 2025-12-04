# LM Studio ACE Integration

Quick start guide for using the ACE framework with LM Studio local models.

## Setup

### 1. Install LM Studio
- Download from [lmstudio.ai](https://lmstudio.ai)
- Install and launch the application

### 2. Download a Model
- In LM Studio, go to the search tab
- Download Gemma 3:1b or another compatible model
- Recommended models:
  - `google/gemma-2-2b-it` (better performance)
  - `microsoft/Phi-3-mini-4k-instruct` (good balance)
  - `google/gemma-2-9b-it` (best performance, requires more RAM)

### 3. Load and Start Server
- Go to the Local Server tab in LM Studio
- Load your downloaded model
- Start the server (default port 1234)
- Verify it's running at `http://localhost:1234`

### 4. Run the Example
```bash
# Install ACE framework if not already installed
pip install ace-framework

# Run the LM Studio starter template
python examples/local-models/lmstudio_starter_template.py
```

## What This Example Demonstrates

1. **Connection Testing**: Verifies LM Studio server is running
2. **Agent Creation**: Sets up ACELiteLLM with LM Studio's OpenAI-compatible API
3. **Before/After Learning**: Shows how the agent improves through ACE training
4. **Strategy Learning**: Demonstrates how the agent learns helpful patterns
5. **Persistence**: Saves learned strategies for future use

## Expected Output

```
✅ LM Studio is running
Loaded model: gemma-2-2b-it

🤖 Creating ACELiteLLM agent with LM Studio...

❓ Testing agent before learning:
Q: What is 2+2?
A: The answer is 4.

🚀 Running ACE learning with LM Studio...
✅ Successfully processed 4/4 samples

📊 Trained on 4 samples
📚 Skillbook now has 3 strategies

🧠 Testing agent after learning:
Q: What is 3+3?
A: 6

💡 Learned strategies:
  1. For arithmetic questions, provide direct numerical answers without... (+1/-0)
  2. When asked about basic facts, respond concisely and accurately (+1/-0)

💾 Saved learned strategies to lmstudio_learned_strategies.json
```

## Troubleshooting

### Common Issues

1. **Connection Error**
   ```
   ❌ Cannot connect to LM Studio. Make sure it's running on port 1234.
   ```
   - Ensure LM Studio server is started
   - Check the port in LM Studio settings
   - Verify no firewall blocking port 1234

2. **JSON Parsing Errors**
   ```
   ❌ Learning failed: JSONDecodeError
   ```
   - Use a larger model (3B+ parameters recommended)
   - Try lowering temperature in the code
   - Check LM Studio console for model errors

3. **Model Not Loaded**
   ```
   ❌ No models loaded in LM Studio
   ```
   - Load a model in LM Studio's Local Server tab
   - Ensure the model is compatible with chat format

### Performance Tips

- **Model Size**: Larger models (7B+) work better with structured JSON output
- **Temperature**: Lower values (0.1-0.3) improve consistency
- **RAM**: Ensure sufficient RAM for your chosen model
- **Batch Size**: Start with small training samples and increase gradually

## Next Steps

1. **Customize Training**: Add your own `Sample` objects for domain-specific learning
2. **Experiment with Models**: Try different models in LM Studio
3. **Persistence**: The agent automatically loads previously learned strategies
4. **Integration**: Use the learned agent in your applications

## Files

- `lmstudio_starter_template.py` - Main example script
- `lmstudio_learned_strategies.json` - Saved strategies (generated after first run)
- `README.md` - This documentation