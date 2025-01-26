from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# LLM model configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("Model loaded.")

# Function to process messages with the LLM
def process_with_llm(user_message: str) -> str:
    try:
        # Structured prompt
        prompt = f"You are a helpful AI assistant. Answer the following question concisely:\n\n{user_message}\n\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        attention_mask = inputs["attention_mask"]  # Add attention mask explicitly
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,  # Pass attention mask
            max_length=100,  # Limit the length of the output
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,  # Avoid padding issues
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract response after "Answer:"
        final_response = response.split("Answer:")[-1].strip()
        return final_response
    except Exception as e:
        return f"Error processing the message: {e}"

# Start function
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hello! I am your AI Assistant. Send me a message, and I'll process it with an LLM!"
    )

# Function to handle text messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and update.message.text:  # Check if the message is text
        user_message = update.message.text
        print(f"Message from {update.message.from_user.username or 'unknown user'}: {user_message}")

        # Process the message with the LLM
        try:
            bot_response = process_with_llm(user_message)
        except Exception as e:
            bot_response = f"Error processing the message: {e}"

        # Reply to the user with the response from the LLM
        await update.message.reply_text(bot_response)

# Bot configuration
def main():
    TOKEN = "7926144412:AAHfvhXD-LIFva3IBKXhYPpVJnn5ggp1ZU0"
    # Create an Application instance
    application = Application.builder().token(TOKEN).build()

    # Add command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()



