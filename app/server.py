import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Add root folder to sys.path to enable clean absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.nn_model import OperatorStrengthNN
from src.features.feature_extractor import OperatorFeatureExtractor
from src.evaluation.evaluator import calculate_gradient_attribution
from app.utils import PROFESSION_METADATA, load_model_and_preprocessor

app = Flask(__name__, template_folder="templates", static_folder="static")

# Pre-load unified neural network model & preprocessor for high performance
unified_model = None
unified_extractor = None

model_path = "models/unified_model.pth"
preprocessor_path = "models/unified_preprocessor.pkl"

print("Pre-loading unified neural network model...")
if os.path.exists(model_path) and os.path.exists(preprocessor_path):
    try:
        unified_extractor = OperatorFeatureExtractor.load(preprocessor_path)
        # Unified model hidden layer dimension is [256, 128, 64]
        unified_model = OperatorStrengthNN(input_dim=unified_extractor.get_input_dim(), hidden_dims=[256, 128, 64])
        unified_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        unified_model.eval()
        print(" -> Unified single model loaded successfully.")
    except Exception as e:
        print(f" -> Failed to load unified model. Error: {str(e)}")
else:
    print(" -> Unified model weights or preprocessor not found. Please run training first.")

@app.route("/")
def index():
    """Renders the main Rhodes Operator Assessment HUD web page."""
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    REST API endpoint for real-time operator cup-tier prediction and attribution using unified model.
    Expects JSON POST payload.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No input data provided"}), 400
            
        if unified_model is None or unified_extractor is None:
            return jsonify({
                "success": False, 
                "error": "Unified model is not loaded or found on server. Please train the model first."
            }), 500
            
        # Extract inputs
        op_name = data.get("name", "未命名干员")
        op_profession = data.get("profession", "Guard")
        op_branch = data.get("branch", "")
        op_type = data.get("type", "物理")
        
        atk = float(data.get("atk", 0))
        df = float(data.get("def", 0))
        res = float(data.get("res", 0))
        dps = float(data.get("dps", 0))
        
        # If Arts damage, DPH is dynamically ignored (set to 0)
        dph = 0.0 if op_type == "法术" else float(data.get("dph", 0))
        
        control_coverage = float(data.get("control_coverage", 0.0))
        buff_amp = float(data.get("buff_amp", 1.0))
        total_damage = float(data.get("total_damage", 0))
        sp_cost = float(data.get("sp_cost", 30))
        init_sp = float(data.get("init_sp", 0))
        
        # Assemble input DataFrame containing 'profession' as a categorical feature
        input_df = pd.DataFrame([{
            "profession": op_profession,
            "atk": atk,
            "def": df,
            "res": res,
            "dps": dps,
            "dph": dph,
            "type": op_type,
            "branch": op_branch,
            "control_coverage": control_coverage,
            "buff_amp": buff_amp,
            "total_damage": total_damage,
            "sp_cost": sp_cost,
            "init_sp": init_sp
        }])
        
        # 1. Transform features (Arts DPH masking happens inside here too)
        preprocessed_input = unified_extractor.transform(input_df)[0]
        
        # 2. PyTorch Prediction
        t_input = torch.tensor(preprocessed_input, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = unified_model(t_input)
            probabilities = torch.softmax(logits, dim=1).numpy()[0]
            
        pred_idx = np.argmax(probabilities)
        pred_cup = unified_extractor.decode_labels(pred_idx)
        pred_prob = float(probabilities[pred_idx])
        
        # Build clean probabilities dictionary
        probs_dict = {
            cup: float(probabilities[unified_extractor.target_mapping[cup]]) 
            for cup in ["小杯", "中杯", "大杯", "超大杯"]
        }
        
        # CRITICAL RULE: Outstanding Niche Tier Bump ("数值明显高于正常则直接升档")
        # If any combat metric is obviously higher/better than the normal average stats, directly bump the cup up by one tier!
        prof_meta = PROFESSION_METADATA[op_profession]
        avg_stats = prof_meta["avg_stats"]
        outstanding_detected = False
        reasons = []

        if atk >= 1.30 * avg_stats["atk"]:
            outstanding_detected = True
            reasons.append("ATK")
        if df >= 1.30 * avg_stats["def"]:
            outstanding_detected = True
            reasons.append("DEF")
        if res >= avg_stats["res"] + 8:
            outstanding_detected = True
            reasons.append("RES")
        if dps >= 1.30 * avg_stats["dps"]:
            outstanding_detected = True
            reasons.append("DPS")
        if op_type != "法术" and dph >= 1.30 * avg_stats["dph"]:
            outstanding_detected = True
            reasons.append("DPH")
        if control_coverage >= avg_stats["control_coverage"] + 0.15:
            outstanding_detected = True
            reasons.append("CC Coverage")
        if buff_amp >= 1.25 * avg_stats["buff_amp"]:
            outstanding_detected = True
            reasons.append("Buff Amp")
        if total_damage >= 1.30 * avg_stats["total_damage"]:
            outstanding_detected = True
            reasons.append("Total Damage")
        if init_sp >= avg_stats["init_sp"] + 10:
            outstanding_detected = True
            reasons.append("Initial SP")
        if sp_cost <= 0.70 * avg_stats["sp_cost"]:
            outstanding_detected = True
            reasons.append("Low SP Cost")

        original_cup = pred_cup
        if outstanding_detected:
            cup_hierarchy = ["小杯", "中杯", "大杯", "超大杯"]
            if pred_cup in cup_hierarchy:
                curr_idx = cup_hierarchy.index(pred_cup)
                if curr_idx < len(cup_hierarchy) - 1:
                    pred_cup = cup_hierarchy[curr_idx + 1]
                    pred_idx = unified_extractor.target_mapping[pred_cup]
                    print(f" -> [Niche Upgrade] Outstanding {reasons} detected. Bumping cup from {original_cup} to {pred_cup}.")
                    
                    # Update probabilities dictionary so UI charts shift dynamically
                    old_prob = probs_dict[original_cup]
                    new_prob = probs_dict[pred_cup]
                    
                    probs_dict[pred_cup] = max(old_prob, 0.75)
                    probs_dict[original_cup] = min(new_prob, 0.20)
                    pred_prob = probs_dict[pred_cup]
        
        # 3. Saliency Attribution Calculation
        attr_df = calculate_gradient_attribution(
            model=unified_model,
            preprocessed_input=preprocessed_input,
            target_class_idx=pred_idx,
            feature_names=unified_extractor.feature_names
        )
        
        # Group and map attribution results for frontend visualization
        agg_attrs = {}
        label_map = {
            "atk": "攻击力 (ATK)", 
            "def": "防御力 (DEF)", 
            "res": "法术抗性 (RES)", 
            "dps": "每秒输出 (DPS)", 
            "dph": "单次破防伤害 (DPH)",
            "control_coverage": "控制覆盖率 (CC)",
            "buff_amp": "辅助增伤幅度 (Buff)",
            "total_damage": "技能总伤 (Total Dmg)",
            "sp_cost": "技能SP消耗 (SP Cost)",
            "init_sp": "初始技力 (Initial SP)"
        }
        
        # Add numeric features
        for num_col, label in label_map.items():
            matched = attr_df[attr_df["feature"] == num_col]
            val = float(matched.iloc[0]["attribution"]) if not matched.empty else 0.0
            agg_attrs[label] = val
            
        # Add categorical type, branch, and profession sums
        type_sum = float(attr_df[attr_df["feature"].str.startswith("type_")]["attribution"].sum())
        branch_sum = float(attr_df[attr_df["feature"].str.startswith("branch_")]["attribution"].sum())
        profession_sum = float(attr_df[attr_df["feature"].str.startswith("profession_")]["attribution"].sum())
        
        agg_attrs["干员属性类型 (Type)"] = type_sum
        agg_attrs["干员分支权重 (Branch)"] = branch_sum
        agg_attrs["职业属性加权 (Profession)"] = profession_sum
        
        # Sort features by absolute contribution
        sorted_attributions = sorted(
            [{"feature": k, "value": v} for k, v in agg_attrs.items()],
            key=lambda x: abs(x["value"]),
            reverse=True
        )
        
        # Retrieve original averages and limits for comparison radar charts
        prof_meta = PROFESSION_METADATA[op_profession]
        
        response_payload = {
            "success": True,
            "operator_name": op_name,
            "predicted_cup": pred_cup,
            "confidence": pred_prob,
            "probabilities": probs_dict,
            "attributions": sorted_attributions,
            "averages": prof_meta["avg_stats"],
            "limits": prof_meta["limits"]
        }
        
        return jsonify(response_payload)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask Rhodes Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
