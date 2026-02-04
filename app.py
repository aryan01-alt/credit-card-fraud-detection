import gradio as gr
import numpy as np
import pickle

# Load the trained model and scaler
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_fraud(time, amount, *features):
    """
    Predict if a transaction is fraudulent
    features: V1 to V28 (PCA components)
    """
    # Combine all inputs
    all_features = [time, *features, amount]
    
    # Convert to numpy array and reshape
    sample = np.array(all_features).reshape(1, -1)
    
    # Scale features
    sample_scaled = scaler.transform(sample)
    
    # Make prediction
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0]
    
    # Format output
    result = "🚨 FRAUD DETECTED" if prediction == 1 else "✅ LEGITIMATE TRANSACTION"
    fraud_prob = probability[1] * 100
    normal_prob = probability[0] * 100
    
    return {
        "Prediction": result,
        "Fraud Probability": f"{fraud_prob:.2f}%",
        "Normal Probability": f"{normal_prob:.2f}%",
        "Risk Level": "HIGH" if fraud_prob > 50 else "LOW"
    }

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Credit Card Fraud Detection") as demo:
    gr.Markdown("# 💳 Credit Card Fraud Detection System")
    gr.Markdown("### Enter transaction details to detect potential fraud")
    
    with gr.Row():
        with gr.Column():
            time_input = gr.Number(label="Time (seconds from first transaction)", value=0)
            amount_input = gr.Number(label="Transaction Amount ($)", value=100.0)
            
            gr.Markdown("### PCA Features (V1-V28)")
            gr.Markdown("*These are anonymized features from PCA transformation*")
            
            # Create inputs for V1-V28
            v_inputs = []
            with gr.Row():
                for i in range(1, 29):
                    if i % 4 == 1 and i > 1:
                        gr.Markdown("")  # Add spacing
                    v_inputs.append(gr.Number(label=f"V{i}", value=0.0, scale=1))
        
        with gr.Column():
            predict_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg")
            output = gr.JSON(label="Analysis Result")
            
            gr.Markdown("---")
            gr.Markdown("### Example Transactions")
            gr.Examples(
                examples=[
                    [406, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100.0],  # Normal
                    [500, -1.5, 1.5, -2.0, 2.5, -1.0, 1.2, -0.8, 0.9, -1.1, 1.3, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 250.0],  # Potential Fraud
                ],
                inputs=[time_input] + v_inputs + [amount_input],
                label="Try these examples"
            )
    
    predict_btn.click(
        fn=predict_fraud,
        inputs=[time_input, amount_input] + v_inputs,
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
