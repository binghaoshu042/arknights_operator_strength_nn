import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from app.utils import inject_arknights_style, load_model_and_preprocessor, PROFESSION_METADATA
from src.evaluation.evaluator import calculate_gradient_attribution

# Set Page Config
st.set_page_config(
    page_title="罗德岛干员杯级神经网络评判系统 v2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set plotting style
sns.set_theme(style="dark")
plt.rcParams['figure.facecolor'] = '#0b0e14'
plt.rcParams['axes.facecolor'] = '#121824'
plt.rcParams['text.color'] = '#d1d5db'
plt.rcParams['axes.labelcolor'] = '#9ca3af'
plt.rcParams['xtick.color'] = '#9ca3af'
plt.rcParams['ytick.color'] = '#9ca3af'

def main():
    # Inject premium custom Rhodes CSS
    inject_arknights_style()
    
    # 1. Header
    st.markdown('<h1 class="rhodes-title">Rhodes Island Operator Evaluation System v2</h1>', unsafe_allow_html=True)
    st.markdown('<p class="rhodes-subtitle">罗德岛干员战术维度神经网络杯级评判系统 (新增总伤、技力、辅助评判)</p>', unsafe_allow_html=True)
    
    # 2. Sidebar Navigation
    st.sidebar.image("https://raw.githubusercontent.com/Aceship/Arknight-Images/main/ui/char_103_amiya_2.png", width=120)
    st.sidebar.markdown("### 🧬 核心数据评判中枢")
    
    # Choose Profession
    prof_keys = list(PROFESSION_METADATA.keys())
    prof_labels = [f"{PROFESSION_METADATA[k]['label']} ({k})" for k in prof_keys]
    
    selected_prof_label = st.sidebar.selectbox("选择干员职业大类:", prof_labels)
    selected_prof = prof_keys[prof_labels.index(selected_prof_label)]
    
    # Get Metadata for selected profession
    meta = PROFESSION_METADATA[selected_prof]
    
    st.sidebar.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid #ff9f1c; padding: 12px; margin-top: 10px;">
        <h4 style="color:#ff9f1c; margin:0 0 5px 0;">{meta['label']} 大类说明</h4>
        <p style="font-size:12px; color:#9ca3af; margin:0;">{meta['desc']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Operator General Info
    op_name = st.sidebar.text_input("干员代号:", value="未命名干员")
    op_branch = st.sidebar.selectbox("干员分支 (Sub-class):", meta["branches"])
    op_type = st.sidebar.selectbox("攻击类型 (Damage Type):", ["物理", "法术", "混合"])
    
    # 3. Main Layout
    # Column 1: Input Stats, Column 2: Neural Network Assessment
    col_input, col_pred = st.columns([1, 1.1])
    
    with col_input:
        st.markdown(f"### 📊 属性参数录入 ({meta['label']})")
        
        lims = meta["limits"]
        avg = meta["avg_stats"]
        
        # Group 1: Base Combat Stats
        st.markdown("#### ⚔️ 基础生存与战斗面板")
        
        col_combat_a, col_combat_b = st.columns(2)
        with col_combat_a:
            atk = st.number_input("攻击力 (ATK):", min_value=lims["atk"][0], max_value=lims["atk"][1], value=avg["atk"], step=10)
            df = st.number_input("防御力 (DEF):", min_value=lims["def"][0], max_value=lims["def"][1], value=avg["def"], step=10)
            res = st.slider("法术抗性 (RES):", min_value=lims["res"][0], max_value=lims["res"][1], value=avg["res"], step=1)
        with col_combat_b:
            dps = st.number_input("每秒输出 (DPS):", min_value=lims["dps"][0], max_value=lims["dps"][1], value=avg["dps"], step=10)
            
            # Arts damage masking logic
            if op_type == "法术":
                st.markdown(f"<div style='background-color:rgba(0, 180, 216, 0.1); border: 1px solid rgba(0, 180, 216, 0.3); border-radius:5px; padding: 7px 10px; margin-bottom: 7px; font-size:12px; color:#00b4d8;'>💡 <strong>法术伤害干员</strong>：无需计算 DPH (物理破防)。系统已将该输入屏蔽并设为 0。</div>", unsafe_allow_html=True)
                dph = 0
            else:
                dph = st.number_input("单次物理伤害 (DPH):", min_value=lims["dph"][0], max_value=lims["dph"][1], value=avg["dph"], step=10)
                
        # Group 2: Tactical & Support Stats
        st.markdown("#### ⚡ 战术技能与辅助面板 (最常用技能)")
        col_tact_a, col_tact_b = st.columns(2)
        with col_tact_a:
            total_damage = st.number_input("技能总伤害/总治疗:", min_value=lims["total_damage"][0], max_value=lims["total_damage"][1], value=avg["total_damage"], step=1000)
            sp_cost = st.slider("技能SP消耗:", min_value=lims["sp_cost"][0], max_value=lims["sp_cost"][1], value=avg["sp_cost"], step=1)
            init_sp = st.slider("初始SP:", min_value=lims["init_sp"][0], max_value=min(lims["init_sp"][1], sp_cost), value=min(avg["init_sp"], sp_cost), step=1)
        with col_tact_b:
            control_coverage = st.slider("控制覆盖率 (Stun/Slow):", min_value=lims["control_coverage"][0], max_value=lims["control_coverage"][1], value=avg["control_coverage"], step=0.05, format="%.2f")
            buff_amp = st.slider("增伤/属性辅助幅度 (Buff):", min_value=lims["buff_amp"][0], max_value=lims["buff_amp"][1], value=avg["buff_amp"], step=0.05, format="%.2f")
            
        # Stat comparison chart (10 attributes)
        st.markdown("#### ⚖️ 10 维战术属性偏离值对比 (Vs. 大类均值)")
        
        categories = ["ATK", "DEF", "RES", "DPS", "DPH", "CC覆盖率", "增伤幅度", "技能总伤", "SP消耗", "初始SP"]
        user_vals = [atk, df, res, dps, dph, control_coverage, buff_amp, total_damage, sp_cost, init_sp]
        avg_vals = [
            avg["atk"], avg["def"], avg["res"], avg["dps"], max(1, avg["dph"]), 
            max(0.01, avg["control_coverage"]), avg["buff_amp"], avg["total_damage"], avg["sp_cost"], max(1, avg["init_sp"])
        ]
        
        # Calculate ratio compared to average
        ratios = []
        for u, a, cat in zip(user_vals, avg_vals, categories):
            if cat == "SP消耗":
                # For SP Cost, lower is better! Calculate inverse ratio
                ratios.append(a / max(1, u) * 100)
            else:
                ratios.append(u / a * 100)
                
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4.0))
        colors = ["#ff9f1c" if r >= 100 else "#00b4d8" for r in ratios]
        
        # For SP consumption, show a label explanation
        display_categories = ["ATK", "DEF", "RES", "DPS", "DPH", "CC", "Buff", "Total Dmg", "SP Cost (Inv)", "Init SP"]
        bars = ax_bar.barh(display_categories, ratios, color=colors, alpha=0.85, height=0.6)
        
        # Add baseline at 100%
        ax_bar.axvline(100, color="#ef4444", linestyle="--", linewidth=1.2, label="Profession Avg (100%)")
        
        # Label bars
        for bar, ratio in zip(bars, ratios):
            width = bar.get_width()
            ax_bar.text(width + 3, bar.get_y() + bar.get_height()/2, f"{ratio:.1f}%", 
                        va='center', ha='left', fontsize=8, color='#d1d5db', fontweight='semibold')
            
        ax_bar.set_xlim(0, max(max(ratios) + 25, 140))
        ax_bar.set_xlabel("Relative Level (%)", fontsize=10)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['left'].set_color('rgba(255,255,255,0.1)')
        ax_bar.spines['bottom'].set_color('rgba(255,255,255,0.1)')
        ax_bar.grid(axis='x', linestyle=':', alpha=0.3)
        ax_bar.legend(loc='lower right', facecolor='#121824', edgecolor='none', labelcolor='#9ca3af', fontsize=8)
        
        st.pyplot(fig_bar)
        
    with col_pred:
        st.markdown("### 🧠 神经网络强度评测")
        
        # Load trained model and preprocessor
        model, extractor = load_model_and_preprocessor(selected_prof)
        
        if model is None or extractor is None:
            st.warning(f"⚠️ 无法加载 [{meta['label']}] 职业的模型文件。请确认您已成功重新训练模型并在 models/ 目录下生成了对应的权重文件。")
            return
            
        # Create input dataframe for transformer
        input_data = pd.DataFrame([{
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
        
        # Preprocess features
        preprocessed_input = extractor.transform(input_data)[0]
        
        # Perform neural network prediction
        t_input = torch.tensor(preprocessed_input, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(t_input)
            probabilities = torch.softmax(logits, dim=1).numpy()[0]
            
        pred_idx = np.argmax(probabilities)
        pred_cup = extractor.decode_labels(pred_idx)
        pred_prob = probabilities[pred_idx]
        
        # Render glowing result badge
        badge_class = "badge-c"
        if pred_cup == "超大杯":
            badge_class = "badge-s"
        elif pred_cup == "大杯":
            badge_class = "badge-a"
        elif pred_cup == "中杯":
            badge_class = "badge-b"
            
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-left: 4px solid {
            '#ef4444' if pred_cup == '超大杯' else ('#f59e0b' if pred_cup == '大杯' else ('#3b82f6' if pred_cup == '中杯' else '#9ca3af'))
        };">
            <h4 style="margin:0; color:#9ca3af; font-size: 14px; text-transform:uppercase;">干员 [{op_name}] 预测评级</h4>
            <div style="margin: 15px 0;">
                <span class="badge {badge_class}" style="font-size: 32px; padding: 8px 24px;">{pred_cup}</span>
            </div>
            <p style="margin:0; font-size:13px; color:#a1a1aa;">神经网络可信度: <strong style="color:#fff;">{pred_prob:.2%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability Bars
        st.markdown("#### 📈 杯级概率概率分布")
        cups = ["小杯", "中杯", "大杯", "超大杯"]
        colors_dict = {"小杯": "#9ca3af", "中杯": "#3b82f6", "大杯": "#f59e0b", "超大杯": "#ef4444"}
        
        for idx, cup in enumerate(cups):
            prob = probabilities[extractor.target_mapping[cup]]
            col_lbl, col_bar = st.columns([1.5, 8.5])
            with col_lbl:
                st.write(f"**{cup}**")
            with col_bar:
                clr = colors_dict[cup]
                st.markdown(f"""
                <div style="background-color: rgba(255,255,255,0.05); border-radius: 5px; width: 100%; height: 18px; margin-top: 3px;">
                    <div style="background-color: {clr}; width: {prob * 100:.1f}%; height: 18px; border-radius: 5px; text-align: right; padding-right: 5px; color: #fff; font-size: 11px; line-height: 18px; font-weight: bold;">
                        {prob * 100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        # 4. Model Explainability / Gradient Saliency Attribution
        st.markdown("#### 🔬 10 维战术维度决策树归因分析 (Gradient Saliency)")
        
        # Compute gradient-based feature attribution
        attr_df = calculate_gradient_attribution(
            model=model, 
            preprocessed_input=preprocessed_input, 
            target_class_idx=pred_idx, 
            feature_names=extractor.feature_names
        )
        
        # Aggregate feature attribution for 10 distinct columns
        agg_attrs = []
        
        label_map = {
            "atk": "攻击力 (ATK)", 
            "def": "防御力 (DEF)", 
            "res": "法术抗性 (RES)", 
            "dps": "每秒输出 (DPS)", 
            "dph": "单次破防伤害 (DPH)",
            "control_coverage": "控制覆盖率 (CC)",
            "buff_amp": "属性/增伤幅度 (Buff)",
            "total_damage": "技能总伤 (Total Dmg)",
            "sp_cost": "技能SP消耗 (SP Cost)",
            "init_sp": "初始技力 (Initial SP)"
        }
        
        for num_col, label in label_map.items():
            matched = attr_df[attr_df["feature"] == num_col]
            if not matched.empty:
                val = matched.iloc[0]["attribution"]
                agg_attrs.append({"Feature": label, "Contribution": val})
                
        # Group categorical type attributes
        type_sum = attr_df[attr_df["feature"].str.startswith("type_")]["attribution"].sum()
        agg_attrs.append({"Feature": "干员属性类型", "Contribution": type_sum})
        
        # Group categorical branch attributes
        branch_sum = attr_df[attr_df["feature"].str.startswith("branch_")]["attribution"].sum()
        agg_attrs.append({"Feature": "干员分支权重", "Contribution": branch_sum})
        
        agg_df = pd.DataFrame(agg_attrs).sort_values(by="Contribution", key=abs, ascending=True)
        
        # Plot local feature attribution
        fig_attr, ax_attr = plt.subplots(figsize=(6, 4.2))
        
        # Generate green for positive contribution, red for negative
        colors = ["#2ec4b6" if c >= 0 else "#e71d36" for c in agg_df["Contribution"]]
        
        bars = ax_attr.barh(agg_df["Feature"], agg_df["Contribution"], color=colors, alpha=0.85, height=0.6)
        
        # Draw center vertical line
        ax_attr.axvline(0, color="gray", linestyle="-", linewidth=0.5)
        
        ax_attr.spines['top'].set_visible(False)
        ax_attr.spines['right'].set_visible(False)
        ax_attr.spines['left'].set_color('rgba(255,255,255,0.1)')
        ax_attr.spines['bottom'].set_color('rgba(255,255,255,0.1)')
        ax_attr.grid(axis='x', linestyle=':', alpha=0.3)
        ax_attr.set_xlabel("Contribution score to predicted class logit", fontsize=10)
        
        st.pyplot(fig_attr)
        st.caption("🟢 绿色代表对当前评级有**正面促进**的战术优势属性；🔴 红色代表对评级起到**扣分拉低**作用的短板属性。")
        
        if op_type == "法术":
            st.info("💡 <strong>法伤屏蔽验证效果</strong>：由于在特征底层将法术类型干员的 DPH 设为 0，上面图中的 <strong>单次破防伤害 (DPH)</strong> 贡献度为完美的 <strong>0.00</strong>，完全不影响神经网络的评估！")

if __name__ == "__main__":
    main()
