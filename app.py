import streamlit as st
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Simple Fine-tuning App",
    page_icon="🤖",
    layout="wide"
)

# App title
st.title("🤖 Simple Transformer Fine-tuning")
st.markdown("Fine-tune DistilBERT on Yelp Reviews with one click!")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Dataset Settings")
    train_size = st.slider("Training samples", 100, 1000, 500)
    test_size = st.slider("Test samples", 50, 500, 100)
    
    st.subheader("Training Settings")
    epochs = st.slider("Epochs", 1, 3, 1)
    batch_size = st.selectbox("Batch size", [8, 16, 32], index=1)
    learning_rate = st.number_input("Learning rate", value=2e-5, format="%.6f")
    
    st.divider()
    output_dir = st.text_input("Output directory", "./fine_tuned_model")

# Main content
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Start Fine-tuning", type="primary", use_container_width=True):
        
        # Step 1: Load dataset
        st.header("📊 Step 1: Loading Yelp Dataset")
        with st.spinner("Loading dataset..."):
            try:
                # Load Yelp Review dataset
                train_dataset = load_dataset("yelp_review_full", split=f"train[:{train_size}]")
                test_dataset = load_dataset("yelp_review_full", split=f"test[:{test_size}]")
                
                st.success(f"✅ Loaded {len(train_dataset)} training and {len(test_dataset)} test samples")
                
                # Show sample
                st.subheader("Sample Review")
                sample = train_dataset[0]
                st.info(f"**Review:** {sample['text'][:200]}...")
                st.info(f"**Rating:** {sample['label'] + 1} stars")  # Yelp uses 1-5, dataset is 0-4
                
                # Show label distribution
                labels = [sample['label'] for sample in train_dataset]
                label_counts = pd.Series(labels).value_counts().sort_index()
                st.bar_chart(label_counts)
                
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                st.stop()
        
        # Step 2: Load model and tokenizer
        st.header("🔧 Step 2: Loading Model")
        with st.spinner("Loading DistilBERT model..."):
            try:
                tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
                model = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert/distilbert-base-uncased",
                    num_labels=5  # Yelp has 5 classes (1-5 stars)
                )
                st.success("✅ Model and tokenizer loaded!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()
        
        # Step 3: Preprocess data
        st.header("🔄 Step 3: Preprocessing")
        with st.spinner("Tokenizing data..."):
            try:
                def tokenize_function(examples):
                    return tokenizer(examples["text"], padding="max_length", truncation=True)
                
                tokenized_train = train_dataset.map(tokenize_function, batched=True)
                tokenized_test = test_dataset.map(tokenize_function, batched=True)
                
                # Format for PyTorch
                tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
                tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])
                
                st.success("✅ Data tokenized and ready!")
            except Exception as e:
                st.error(f"Error preprocessing data: {e}")
                st.stop()
        
        # Step 4: Training
        st.header("🏃‍♂️ Step 4: Training")
        with st.spinner("Training model (this may take a few minutes)..."):
            try:
                # Create output directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Training arguments
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    learning_rate=learning_rate,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    logging_dir=f'{output_dir}/logs',
                    load_best_model_at_end=True,
                    metric_for_best_model="accuracy",
                    greater_is_better=True,
                    report_to="none",
                    save_total_limit=1,
                )
                
                # Compute metrics function
                def compute_metrics(eval_pred):
                    logits, labels = eval_pred
                    predictions = np.argmax(logits, axis=-1)
                    accuracy_metric = evaluate.load("accuracy")
                    return accuracy_metric.compute(predictions=predictions, references=labels)
                
                # Create trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_train,
                    eval_dataset=tokenized_test,
                    compute_metrics=compute_metrics,
                )
                
                # Train!
                train_result = trainer.train()
                eval_result = trainer.evaluate()
                
                # Save model
                trainer.save_model(output_dir)
                
                st.success("🎉 Training completed!")
                
                # Display results
                st.subheader("📈 Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Loss", f"{train_result.training_loss:.4f}")
                with col2:
                    st.metric("Evaluation Loss", f"{eval_result['eval_loss']:.4f}")
                with col3:
                    st.metric("Accuracy", f"{eval_result['eval_accuracy']:.4f}")
                
            except Exception as e:
                st.error(f"Error during training: {e}")
                st.stop()
        
        # Step 5: Test the model
        st.header("🔍 Step 5: Test the Model")
        
        test_review = st.text_area(
            "Enter a review to test:",
            value="The food was absolutely delicious and the service was excellent!",
            height=100
        )
        
        if st.button("Predict Rating"):
            with st.spinner("Predicting..."):
                try:
                    # Load the fine-tuned model
                    fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(output_dir)
                    
                    # Tokenize input
                    inputs = tokenizer(test_review, return_tensors="pt", truncation=True, padding=True)
                    
                    # Make prediction
                    with torch.no_grad():
                        outputs = fine_tuned_model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    
                    # Get predicted class (add 1 since Yelp uses 1-5 stars)
                    predicted_class = torch.argmax(predictions).item() + 1
                    
                    st.success(f"**Predicted Rating: {predicted_class} stars**")
                    
                    # Show probabilities
                    st.subheader("Star Probabilities:")
                    probs_df = pd.DataFrame({
                        "Stars": ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                        "Probability": predictions[0].numpy()
                    })
                    
                    # Format probabilities as percentages
                    probs_df["Probability %"] = (probs_df["Probability"] * 100).round(2).astype(str) + "%"
                    
                    st.dataframe(
                        probs_df[["Stars", "Probability %"]],
                        width=400,
                        hide_index=True
                    )
                    
                    # Simple bar chart
                    st.bar_chart(probs_df.set_index("Stars")["Probability"])
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# Footer
st.divider()
st.caption("Built with ❤️ using Streamlit and Hugging Face Transformers")
