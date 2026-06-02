import os
import random
import pandas as pd
import numpy as np

def create_mock_data(output_path="data/raw/operators.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 8 Professions and their configuration structures (DPH expanded to 15,000)
    professions = {
        "Guard": {
            "branches": ["强攻手", "领主", "无畏者", "剑豪", "术战者", "收割者", "解放者"],
            "types": ["物理", "法术", "混合"],
            "atk_range": (700, 2000),
            "def_range": (300, 650),
            "res_range": (0, 25),
            "dps_range": (500, 5000),
            "dph_range": (600, 15000),
            "cc_range": (0.0, 0.4),
            "buff_range": (1.0, 1.3),
            "td_range": (20000, 300000),
            "sp_range": (30, 80),
            "init_sp_range": (0, 50)
        },
        "Vanguard": {
            "branches": ["尖兵", "冲锋手", "战术家", "情报官", "执旗手"],
            "types": ["物理", "混合"],
            "atk_range": (350, 1000),
            "def_range": (300, 550),
            "res_range": (0, 15),
            "dps_range": (200, 2000),
            "dph_range": (300, 8000),
            "cc_range": (0.0, 0.3),
            "buff_range": (1.0, 1.2),
            "td_range": (5000, 100000),
            "sp_range": (10, 45),
            "init_sp_range": (0, 30)
        },
        "Defender": {
            "branches": ["守护者", "决战铁卫", "驭法铁卫", "不屈者", "要塞"],
            "types": ["物理", "法术", "混合"],
            "atk_range": (400, 1500),
            "def_range": (650, 1400),
            "res_range": (0, 35),
            "dps_range": (100, 3000),
            "dph_range": (200, 10000),
            "cc_range": (0.0, 0.7),
            "buff_range": (1.0, 1.3),
            "td_range": (5000, 150000),
            "sp_range": (15, 60),
            "init_sp_range": (0, 40)
        },
        "Sniper": {
            "branches": ["速射手", "轰击手", "投掷手", "重射手", "战术射手", "散布手"],
            "types": ["物理"],
            "atk_range": (600, 2200),
            "def_range": (120, 300),
            "res_range": (0, 20),
            "dps_range": (600, 6000),
            "dph_range": (500, 15000),
            "cc_range": (0.0, 0.2),
            "buff_range": (1.0, 1.1),
            "td_range": (25000, 350000),
            "sp_range": (20, 60),
            "init_sp_range": (0, 40)
        },
        "Caster": {
            "branches": ["中坚术师", "扩散术师", "阵法术师", "驭械术师", "轰击术师", "秘术师"],
            "types": ["法术"],
            "atk_range": (700, 2200),
            "def_range": (100, 250),
            "res_range": (15, 40),
            "dps_range": (400, 5000),
            "dph_range": (700, 15000),
            "cc_range": (0.0, 0.5),
            "buff_range": (1.0, 1.2),
            "td_range": (25000, 300000),
            "sp_range": (30, 80),
            "init_sp_range": (0, 50)
        },
        "Medic": {
            "branches": ["医师", "群愈师", "疗养师", "行医", "咒愈师"],
            "types": ["法术", "物理"],
            "atk_range": (400, 1500),
            "def_range": (120, 350),
            "res_range": (0, 30),
            "dps_range": (200, 2500),
            "dph_range": (300, 10000),
            "cc_range": (0.0, 0.1),
            "buff_range": (1.0, 2.0),
            "td_range": (15000, 150000),
            "sp_range": (15, 70),
            "init_sp_range": (0, 50)
        },
        "Supporter": {
            "branches": ["凝滞师", "削弱者", "吟游诗人", "召唤师", "护佑者", "工匠"],
            "types": ["法术", "混合"],
            "atk_range": (450, 1500),
            "def_range": (120, 350),
            "res_range": (15, 35),
            "dps_range": (200, 3000),
            "dph_range": (300, 10000),
            "cc_range": (0.2, 1.0),
            "buff_range": (1.1, 2.5),
            "td_range": (5000, 150000),
            "sp_range": (10, 60),
            "init_sp_range": (0, 45)
        },
        "Specialist": {
            "branches": ["处决者", "推击手", "钩手", "行商", "陷阱师", "傀儡师", "伏击客"],
            "types": ["物理", "法术", "混合"],
            "atk_range": (550, 1800),
            "def_range": (200, 550),
            "res_range": (0, 30),
            "dps_range": (400, 5000),
            "dph_range": (500, 15000),
            "cc_range": (0.1, 1.0),
            "buff_range": (1.0, 1.8),
            "td_range": (15000, 250000),
            "sp_range": (5, 50),
            "init_sp_range": (0, 35)
        }
    }
    
    random.seed(42)
    np.random.seed(42)
    
    operators_list = []
    
    for prof_name, cfg in professions.items():
        for i in range(105): # Generate 105 per profession -> total ~840
            branch = random.choice(cfg["branches"])
            op_type = random.choice(cfg["types"])
            
            # Base stats with noise
            atk = int(np.random.normal((cfg["atk_range"][0] + cfg["atk_range"][1]) / 2, (cfg["atk_range"][1] - cfg["atk_range"][0]) / 6))
            atk = max(cfg["atk_range"][0], min(cfg["atk_range"][1], atk))
            
            df = int(np.random.normal((cfg["def_range"][0] + cfg["def_range"][1]) / 2, (cfg["def_range"][1] - cfg["def_range"][0]) / 6))
            df = max(cfg["def_range"][0], min(cfg["def_range"][1], df))
            
            res = random.randint(cfg["res_range"][0], cfg["res_range"][1])
            
            dps = int(np.random.normal((cfg["dps_range"][0] + cfg["dps_range"][1]) / 2, (cfg["dps_range"][1] - cfg["dps_range"][0]) / 6))
            dps = max(cfg["dps_range"][0], min(cfg["dps_range"][1], dps))
            
            dph = int(np.random.normal((cfg["dph_range"][0] + cfg["dph_range"][1]) / 2, (cfg["dph_range"][1] - cfg["dph_range"][0]) / 6))
            dph = max(cfg["dph_range"][0], min(cfg["dph_range"][1], dph))
            
            # Tactical properties (continuous)
            control_coverage = float(np.random.uniform(cfg["cc_range"][0], cfg["cc_range"][1]))
            buff_amp = float(np.random.uniform(cfg["buff_range"][0], cfg["buff_range"][1]))
            total_damage = int(np.random.uniform(cfg["td_range"][0], cfg["td_range"][1]))
            sp_cost = int(np.random.uniform(cfg["sp_range"][0], cfg["sp_range"][1]))
            
            # Make sure init_sp is not greater than sp_cost
            init_sp = int(np.random.uniform(cfg["init_sp_range"][0], min(cfg["init_sp_range"][1], sp_cost)))
            
            score = 0.0
            
            # Smart scoring logic based on expanded limits and DPH = 15,000 scale
            if prof_name == "Guard":
                if op_type == "法术":
                    # Arts Guard completely ignores DPH
                    score = (dps / 5000) * 0.35 + (total_damage / 300000) * 0.35 + (atk / 2000) * 0.10 + (df / 650) * 0.10 + (res / 25) * 0.05
                else:
                    # Physical / Hybrid Guard values DPH highly (DPH up to 15,000 scale)
                    score = (dps / 5000) * 0.20 + (dph / 15000) * 0.15 + (total_damage / 300000) * 0.35 + (atk / 2000) * 0.10 + (df / 650) * 0.10 + (res / 25) * 0.05
                if branch in ["解放者", "术战者", "领主"]:
                    score += 0.02
                    
            elif prof_name == "Vanguard":
                # High init_sp/sp_cost represents extremely fast deploy recovery (like Myrtle)
                startup_ratio = init_sp / max(1, sp_cost)
                score = (dps / 2000) * 0.10 + (df / 550) * 0.10 + (res / 15) * 0.05 + (atk / 1000) * 0.10 + (startup_ratio) * 0.40 + (total_damage / 100000) * 0.20
                if branch in ["执旗手", "情报官"]:
                    score += 0.02
                    
            elif prof_name == "Defender":
                # Defenders care heavily about def, res, control and total damage
                score = (df / 1400) * 0.35 + (res / 35) * 0.15 + (control_coverage) * 0.25 + (total_damage / 150000) * 0.20
                if branch in ["不屈者", "守护者"]:
                    score += 0.02
                    
            elif prof_name == "Sniper":
                # Snipers care about DPS, DPH (Lemuen physical burst), total damage, defense/res
                score = (dps / 6000) * 0.20 + (dph / 15000) * 0.15 + (total_damage / 350000) * 0.40 + (atk / 2200) * 0.10 + (df / 300) * 0.05 + (res / 20) * 0.05
                
            elif prof_name == "Caster":
                # Casters ignore DPH. Care heavily about total damage, dps, res and slight def value
                score = (dps / 5000) * 0.25 + (total_damage / 300000) * 0.40 + (res / 40) * 0.15 + (atk / 2200) * 0.10 + (df / 250) * 0.05
                if branch in ["阵法术师", "中坚术师"]:
                    score += 0.02
                    
            elif prof_name == "Medic":
                # Medics value healing capacity (atk, dps, total_healing), buffing (buff_amp) and defense/res
                score = (atk / 1500) * 0.20 + (dps / 2500) * 0.10 + (total_damage / 150000) * 0.20 + ((buff_amp - 1.0) / 1.0) * 0.35 + (df / 350) * 0.08 + (res / 30) * 0.07
                if branch in ["咒愈师", "行医"]:
                    score += 0.02
                    
            elif prof_name == "Supporter":
                # Supporters are judged heavily by buff_amp, control_coverage, res and defense
                score = ((buff_amp - 1.0) / 1.5) * 0.40 + (control_coverage) * 0.30 + (res / 35) * 0.10 + (total_damage / 150000) * 0.10 + (df / 350) * 0.10
                if branch in ["召唤师", "吟游诗人"]:
                    score += 0.02
                    
            elif prof_name == "Specialist":
                # Specialists value low sp_cost, dps, control and survivability
                sp_factor = max(0, 1.0 - (sp_cost / 35))
                score = (dps / 5000) * 0.20 + (total_damage / 250005) * 0.20 + (control_coverage) * 0.25 + (sp_factor) * 0.20 + (df / 550) * 0.10 + (res / 30) * 0.05
                if branch in ["处决者"]:
                    score += 0.02
            
            # CRITICAL RULE: Numerical niche override ("极值就业空间加分")
            # If any key tactical metric is extremely high, the operator has a guaranteed high niche/employment space, so they are boosted to at least 大杯 or 超大杯.
            norm_dps = (dps - cfg["dps_range"][0]) / (cfg["dps_range"][1] - cfg["dps_range"][0])
            norm_dph = (dph - cfg["dph_range"][0]) / (cfg["dph_range"][1] - cfg["dph_range"][0]) if op_type != "法术" else 0.0
            norm_td = (total_damage - cfg["td_range"][0]) / (cfg["td_range"][1] - cfg["td_range"][0])
            norm_cc = (control_coverage - cfg["cc_range"][0]) / (cfg["cc_range"][1] - cfg["cc_range"][0]) if cfg["cc_range"][1] > cfg["cc_range"][0] else 0.0
            norm_buff = (buff_amp - cfg["buff_range"][0]) / (cfg["buff_range"][1] - cfg["buff_range"][0]) if cfg["buff_range"][1] > cfg["buff_range"][0] else 0.0
            norm_def = (df - cfg["def_range"][0]) / (cfg["def_range"][1] - cfg["def_range"][0])
            norm_sp_startup = (init_sp / max(1, sp_cost))
            norm_low_sp = (cfg["sp_range"][1] - sp_cost) / (cfg["sp_range"][1] - cfg["sp_range"][0])

            max_tactical_val = max(
                norm_dps, 
                norm_dph, 
                norm_td, 
                norm_cc, 
                norm_buff,
                norm_low_sp,
                norm_def if prof_name == "Defender" else 0.0,
                norm_sp_startup if prof_name == "Vanguard" else 0.0
            )
            
            if max_tactical_val >= 0.85:
                # If a stat is at the absolute peak, it secures a powerful niche, guaranteeing at least "大杯" (floor 0.58) and scaling into "超大杯" (up to 0.80)
                niche_score = 0.58 + 0.22 * max_tactical_val
                score = max(score, niche_score)
            
            # Map score to cup classes with some random variance (10% noise)
            score += np.random.normal(0, 0.04)
            
            if score >= 0.72:
                cup = "超大杯"
            elif score >= 0.53:
                cup = "大杯"
            elif score >= 0.36:
                cup = "中杯"
            else:
                cup = "小杯"
                
            name = f"{prof_name}_{branch}_{i+1:03d}"
            
            operators_list.append({
                "name": name,
                "profession": prof_name,
                "branch": branch,
                "atk": atk,
                "def": df,
                "res": res,
                "dps": dps,
                "dph": dph,
                "type": op_type,
                "control_coverage": round(control_coverage, 3),
                "buff_amp": round(buff_amp, 2),
                "total_damage": total_damage,
                "sp_cost": sp_cost,
                "init_sp": init_sp,
                "cup": cup
            })
            
    df_all = pd.DataFrame(operators_list)
    df_all.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Generated {len(df_all)} power creep mock operators with DPH limits up to 15,000 saved to {output_path}")
    print("Class distribution:")
    print(df_all['cup'].value_counts())

if __name__ == "__main__":
    create_mock_data()
