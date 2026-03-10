"""
AI 阿瓦隆（The Resistance: Avalon）
====================================
5 人局，角色固定為：梅林×1、刺客×1、爪牙×1、忠臣×2
玩家 1 為人類，玩家 2–5 由 AI 控制。

架構：
  - 控管 AI  (controller)  ── 判斷遊戲流程、發出 CALL 指令
  - 主持人 AI (host)        ── 將控管 AI 的輸出包裝成自然語言
  - 玩家 AI  (player 2–5)  ── 各自扮演角色並做出策略回應
"""

# ─────────────────────────────────────────────
# 1. 套件匯入
# ─────────────────────────────────────────────
import os
import re
import json
import random
from collections import Counter
import gradio as gr
import aisuite as ai_suite
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# 2. 環境變數與 API 設定
# ─────────────────────────────────────────────
load_dotenv("../Data/AI_code.env")

#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
#PROVIDER = "openai"
#MODEL    = "gpt-4o"
PROVIDER = "groq"
MODEL    = "llama-3.3-70b-versatile"
client = ai_suite.Client()

# ─────────────────────────────────────────────
# 3. 遊戲常數
# ─────────────────────────────────────────────
ROLE_ABILITIES: dict[str, str] = {
    "梅林": "你知道所有邪惡陣營成員，但不能暴露自己的身份。",
    "刺客": "你知道所有邪惡陣營成員，並可在遊戲結束時刺殺你認為是梅林的玩家。",
    "爪牙": "你知道其他邪惡陣營成員（刺客與爪牙），你要幫助邪惡方完成任務失敗。",
    "忠臣": "你不知道其他人的身份，請協助正義方完成三次任務成功。",
}

AVAILABLE_ROLES = Counter({"梅林": 1, "刺客": 1, "爪牙": 1, "忠臣": 2})

# 五人局每輪出任務人數
TEAM_SIZES = [2, 3, 2, 3, 3]

# ─────────────────────────────────────────────
# 4. System Prompts
# ─────────────────────────────────────────────
CONTROLLER_PROMPT = '''
你是阿瓦隆桌遊的控管 AI，負責根據遊戲狀態 (game_state) 和輸入，決定要執行什麼步驟。
你不能直接改變遊戲狀態，而是要用 CALL 指令讓系統執行。

game_state 說明：
- players：所有玩家的名字（順序即代表順時針方向）
- roles：目前每位玩家的角色（遊戲開始前為空）
- available_roles：可使用的角色及其數量
- leader_index：目前輪到哪一位玩家指派任務（以索引位置表示）
- round：目前遊戲進行到第幾回合
- fail_count / success_count：任務失敗/成功的累計次數
- current_phase：目前遊戲所處階段
- team_proposal：本輪由隊長提名出任務的玩家列表
- team_votes / mission_votes：所有玩家的投票紀錄
- discussion_log：玩家對話紀錄
- assassination_target：刺客階段時目標玩家名
- winner：遊戲結束後獲勝方
- team_sizes：每一輪應派出幾位玩家 [2, 3, 2, 3, 3]

CALL 指令格式：
- CALL: reset_game
- CALL: assign_roles
- CALL: update_phase(phase="任務指派")
- CALL: advance_leader
- CALL: record_team_votes(玩家1="同意", 玩家2="反對", ...)
- CALL: record_team_proposal(玩家2, 玩家4)
- CALL: record_mission_votes(玩家3="成功", 玩家4="失敗")
- CALL: increment_fail_count
- CALL: increment_success_count
- CALL: increment_round
- CALL: set_assassination_target(玩家名)
- CALL: set_winner("正義") / set_winner("邪惡")

流程順序（嚴禁跳過）：
1. 分派角色 → 2. 任務指派 → 3. 隊伍投票 → 4. 任務進行
→ 5. 任務結束後討論（僅任務失敗時，三輪）→ 6. 判斷結束／下一輪
→ 7. 刺殺階段（任務成功三次後）→ 8. 宣布勝負

出任務人數必須讀取：game_state["team_sizes"][game_state["round"] - 1]
絕對不可硬寫數字。

隊伍投票邏輯：
- 贊成 > 反對 → CALL: update_phase(phase="任務進行")
- 否則 → CALL: advance_leader 然後 CALL: update_phase(phase="任務指派")

任務結束邏輯：
- 有失敗票 → increment_fail_count → update_phase(phase="任務結束後討論")
- 全成功票 → increment_success_count → 若 success_count==3 進刺殺，否則 increment_round + advance_leader + 下一輪任務指派

阿瓦隆規則：
- 連續五輪提案被否決，第五輪自動出任務
- 失敗三次 → 邪惡方勝利
- 成功三次 → 刺殺階段；若刺中梅林 → 邪惡方勝，否則正義方勝

處理完 CALL 指令後，請補上一句玩家可理解的引導語句，例如：
- 「請玩家3提名2位玩家出任務。」
- 「這次任務失敗，請開始討論。」

【強制規定】
在任務指派階段，隊長提名玩家後，你必須先執行：
CALL: record_team_proposal(玩家X, 玩家Y, ...)
確認提名人數等於 game_state["team_sizes"][game_state["round"] - 1] 之後，
才能執行 CALL: update_phase(phase="隊伍投票")
如果人數不對，必須要求隊長重新提名，不可以直接進入投票階段。

在處理任何投票或決策之前，必須先輸出你對玩家輸入的解讀，例如：
- 「玩家1的回應解讀為：反對」
- 「玩家1的回應解讀為：任務成功」
- 「玩家1提名的隊員解讀為：玩家2、玩家4」

如果玩家的輸入完全無法判斷意圖（例如空白、或與當前階段無關），
不要推進遊戲，輸出：
「玩家1的回應無法解讀，請重新輸入。」

'''

PLAYER_PROMPT = '''你是一位參與《阿瓦隆》桌上遊戲的玩家。
根據你的角色身份與主持人訊息，做出符合陣營利益的回應。

規則重點：
- 正義方（梅林、忠臣）讓任務成功；邪惡方（刺客、爪牙）讓任務失敗
- 玩家無法公開知道他人角色，但邪惡方互知，梅林知道邪惡方

行為準則：
- 投票階段：直接回覆結果，例如「我同意這組人出任務。」
- 任務階段：直接回覆「我選擇任務成功。」或「我選擇任務失敗。」
- 討論階段：針對遊戲狀況給出有邏輯的發言
- 邪惡方：不要在發言中洩漏自己的陣營或意圖
- 正義方：在任務階段只能回覆「我選擇任務成功。」
- 隊長指派：不可重複提名同一位玩家

輸出格式：只輸出你的回應內容，不要加分析說明。
'''

HOST_PROMPT = '''你是《阿瓦隆》桌遊的主持人 AI。
將控管 AI 的輸出包裝成自然、有臨場感的主持人語氣。

輸出格式（固定 5 筆 JSON 陣列，不要加 ```json 標記）：
[
  { "target": "player_1", "content": "..." },
  { "target": "player_2", "content": "..." },
  { "target": "player_3", "content": "..." },
  { "target": "player_4", "content": "..." },
  { "target": "player_5", "content": "..." }
]

行為準則：
- 不提及 CALL、function、update_phase 等技術詞彙
- 若該玩家無需收到訊息，content 設為 ""
- 任務指派階段：只通知隊長，其他玩家 content 留空
- 進入隊伍投票階段時，必須明確告知所有玩家：本輪隊長是誰、隊長提名了哪些人出任務。
  例如：「本輪隊長是玩家2，他提名了玩家2和玩家3出任務，請大家進行投票。」
- 宣告任務結果時：只說成功或失敗，絕對不透露誰投了什麼
- 單人決策（指派、任務投票、刺殺）只分配給對應玩家
- 用台灣習慣的中文，語氣親切自然
- 宣告任務結果時：必須提及本輪出任務的玩家是誰，
  從 game_state["team_proposal"] 取得名單，
  例如「這次出任務的是玩家2和玩家5，任務失敗了，現在進入討論階段。」

【強制規定】
分派角色階段：所有玩家的 content 一律留空 ""，不需要告知任何人任何事。
私人訊息由系統負責發送，主持人不處理這個階段。
'''

# ─────────────────────────────────────────────
# 5. 遊戲狀態初始化
# ─────────────────────────────────────────────
def make_game_state() -> dict:
    return {
        "players":    ["玩家1", "玩家2", "玩家3", "玩家4", "玩家5"],
        "roles":      {},
        "available_roles": dict(AVAILABLE_ROLES),
        "leader_index": random.randint(0, 4),
        "round":      1,
        "fail_count": 0,
        "success_count": 0,
        "current_phase": "遊戲開始",
        "team_proposal": [],
        "team_votes":  {},
        "mission_votes": {},
        "discussion_log": [],
        "assassination_target": None,
        "winner":     None,
        "team_sizes": TEAM_SIZES[:],
    }

def make_session_state() -> dict:
    """儲存各 AI 的訊息歷史與 prompt，以及討論輪次計數。"""
    return {
        "discussion_count": 1,
        "p2_msg": "", "p3_msg": "", "p4_msg": "", "p5_msg": "",
        "p2_prompt": "", "p3_prompt": "", "p4_prompt": "", "p5_prompt": "",
        "p2_messages": [], "p3_messages": [], "p4_messages": [], "p5_messages": [],
        "controller_messages": [],
        "host_messages": [],
        "pending_discussion": {},   # 討論結束後暫存，等下一輪一起給玩家
    }

def initialize() -> tuple[dict, dict]:
    return make_game_state(), make_session_state()

# ─────────────────────────────────────────────
# 6. CALL 指令處理
# ─────────────────────────────────────────────
def _parse_vote_dict(call_str: str) -> dict:
    return {name: vote for name, vote in re.findall(r'(玩家\d+)\s*=\s*"(.*?)"', call_str)}

def process_call(call: str, gs: dict) -> None:
    """根據 CALL 指令更新 game_state（in-place）。"""
    if call.startswith("reset_game"):
        gs.update(make_game_state())

    elif call.startswith("assign_roles"):
        role_list = [role for role, cnt in gs["available_roles"].items() for _ in range(cnt)]
        random.shuffle(role_list)
        gs["roles"] = dict(zip(gs["players"], role_list))

    elif call.startswith("update_phase"):
        m = re.search(r'phase\s*=\s*"(.*?)"', call)
        if m:
            gs["current_phase"] = m.group(1)

    elif call.startswith("advance_leader"):
        gs["leader_index"] = (gs["leader_index"] + 1) % len(gs["players"])

    elif call.startswith("record_team_votes"):
        gs["team_votes"] = _parse_vote_dict(call)

    elif call.startswith("record_mission_votes"):
        gs["mission_votes"] = _parse_vote_dict(call)

    elif call.startswith("increment_fail_count"):
        gs["fail_count"] += 1

    elif call.startswith("increment_success_count"):
        gs["success_count"] += 1

    elif call.startswith("increment_round"):
        gs["round"] += 1

    elif call.startswith("set_assassination_target"):
        m = re.search(r'\((.*?)\)', call)
        if m:
            gs["assassination_target"] = m.group(1)

    elif call.startswith("set_winner"):
        m = re.search(r'\("?(.*?)"?\)', call)
        if m:
            gs["winner"] = m.group(1)

    elif call.startswith("record_team_proposal"):
        candidates = re.findall(r"玩家\d+", call)
        required = gs["team_sizes"][gs["round"] - 1]
        # 有重複或人數不對就清空，讓隊長重新指派
        gs["team_proposal"] = candidates if len(candidates) == required and len(set(candidates)) == required else []

# ─────────────────────────────────────────────
# 7. AI 呼叫封裝
# ─────────────────────────────────────────────
def call_ai(system: str, prompt: str, messages: list,
            provider: str = PROVIDER, model: str = MODEL) -> tuple[str, list]:
    """呼叫 AI，回傳 (回應文字, 更新後的 messages)。"""
    if not messages:
        messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=f"{provider}:{model}", messages=messages)
    content = response.choices[0].message.content
    messages.append({"role": "assistant", "content": content})
    return content, messages

def run_controller(gs: dict, player_inputs: dict, messages: list) -> tuple[str, list, dict]:
    """執行控管 AI，解析並套用所有 CALL 指令，回傳 (回應文字, messages, 更新後 gs)。"""
    prompt = (
        f"目前遊戲狀態：\n{json.dumps(gs, ensure_ascii=False, indent=2)}\n\n"
        f"玩家輸入：\n" +
        "".join(f"- {p}：{r}\n" for p, r in player_inputs.items()) +
        "\n請回傳要執行的 CALL 指令（每個獨立一行），並附上引導語句。"
    )
    reply, messages = call_ai(CONTROLLER_PROMPT, prompt, messages)
    for line in reply.splitlines():
        if line.strip().startswith("CALL:"):
            process_call(line.strip().removeprefix("CALL:").strip(), gs)
    if gs["current_phase"] == "任務指派" and gs["team_proposal"] == []:
        # team_proposal 被清空代表人數不對，補一個提示讓控管 AI 重來
        required = gs["team_sizes"][gs["round"] - 1]
        leader = gs["players"][gs["leader_index"]]
        fix_prompt = f"提名人數不正確，第{gs['round']}輪需要{required}人，請{leader}重新提名{required}位不重複的玩家。"
        reply2, messages = call_ai(CONTROLLER_PROMPT, fix_prompt, messages)
        for line in reply2.splitlines():
            if line.strip().startswith("CALL:"):
                process_call(line.strip().removeprefix("CALL:").strip(), gs)
        return reply2, messages, gs
    return reply, messages, gs

def run_host(controller_reply: str, messages: list) -> tuple[list[dict], list]:
    """執行主持人 AI，將控管輸出包裝成 JSON 格式的玩家訊息。"""
    raw, messages = call_ai(HOST_PROMPT, controller_reply, messages)
    try:
        return json.loads(raw), messages
    except json.JSONDecodeError:
        # 容錯：嘗試抽取 [...] 區塊
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            return json.loads(m.group()), messages
        raise

# ─────────────────────────────────────────────
# 8. 私人訊息產生
# ─────────────────────────────────────────────
def build_private_messages(gs: dict) -> dict[str, str]:
    """回傳 {player_N: 私人訊息} 的字典。"""
    roles = gs["roles"]
    players = gs["players"]
    evil = [p for p in players if roles[p] in ("刺客", "爪牙")]
    private = {}
    for i, player in enumerate(players):
        role = roles[player]
        msg = f"你是{role}，{ROLE_ABILITIES[role]}"
        if role == "梅林":
            msg += f" 你知道 {', '.join(evil)} 是邪惡陣營，但不能暴露自己的身份。"
        elif role in ("刺客", "爪牙"):
            teammates = [p for p in evil if p != player]
            if teammates:
                msg += f" 你知道 {', '.join(teammates)} 為邪惡陣營，請互相配合讓任務失敗。"
        private[f"player_{i+1}"] = msg
    return private

# ─────────────────────────────────────────────
# 9. 主遊戲迴圈
# ─────────────────────────────────────────────
def _host_to_map(host_output: list[dict]) -> dict[str, str]:
    return {entry["target"]: entry["content"] for entry in host_output}

def game_step(user_input: str, gs: dict, ss: dict) -> tuple[str, dict, dict]:
    """
    處理玩家 1 的一次輸入，推進遊戲狀態，回傳：
      (player_1 看到的訊息, 更新後 gs, 更新後 ss)

    討論階段邏輯：
      - 前兩輪（discussion_count < 3）：蒐集發言，顯示給玩家 1
      - 第三輪（discussion_count == 3）：蒐集完畢後送入控管 AI 結束討論，
        將討論摘要暫存在 pending_discussion，下次需要玩家行動時一併顯示
    """
    # ── 解包 session state ──
    p_msg    = {k: ss[f"p{k}_msg"]      for k in (2, 3, 4, 5)}
    p_prompt = {k: ss[f"p{k}_prompt"]   for k in (2, 3, 4, 5)}
    p_msgs   = {k: ss[f"p{k}_messages"] for k in (2, 3, 4, 5)}
    ctrl_msgs = ss["controller_messages"]
    host_msgs = ss["host_messages"]
    disc_cnt  = ss["discussion_count"]
    pending   = ss["pending_discussion"]

    in_discussion = "討論" in gs["current_phase"]
    private = {f"player_{i+1}": "" for i in range(5)}
    # ── 取得 AI 玩家回覆 ──
    def get_reply(k: int) -> str:
        if not p_msg[k]:
            return ""
        r, p_msgs[k] = call_ai(PLAYER_PROMPT, p_prompt[k], p_msgs[k])
        return r

    p_reply = {k: get_reply(k) for k in (2, 3, 4, 5)}

    # ── 討論階段：蒐集發言 ──
    if in_discussion:
        # 每位玩家看到的討論記錄（排除自己）
        all_replies = {1: user_input, **p_reply}
        round_log = " \n".join(f"玩家{n}:{all_replies[n]}" for n in range(1, 6) if all_replies[n])

        if disc_cnt < 3:
            # 累積討論，直接顯示給 player_1
            disc_cnt += 1
            # 更新各玩家 prompt（加入本輪其他人的發言）
            for k in (2, 3, 4, 5):
                others = " \n".join(f"玩家{n}:{all_replies[n]}" for n in range(1, 6) if n != k)
                p_prompt[k] = others
                p_msg[k] = p_prompt[k]

            player1_view = round_log

        else:
            # 第三輪：結束討論，送控管 AI
            disc_cnt = 1
            end_input = {p: "討論結束" for p in gs["players"]}
            ctrl_reply, ctrl_msgs, gs = run_controller(gs, end_input, ctrl_msgs)
            host_output, host_msgs = run_host(ctrl_reply, host_msgs)
            msg_map = _host_to_map(host_output)

            if gs["roles"]:
                private = build_private_messages(gs)

            # 組合：私人訊息 + 討論摘要 + 主持人訊息
            disc_summary = round_log
            pending = {
                f"player_{k}": " \n".join(f"玩家{n}:{all_replies[n]}" for n in range(1, 6) if n != k)
                for k in range(1, 6)
            }
            for k in (2, 3, 4, 5):
                p_prompt[k] = "\n".join(filter(None, [private[f"player_{k}"], pending[f"player_{k}"], msg_map[f"player_{k}"]]))
                p_msg[k] = msg_map[f"player_{k}"]
            player1_view = "\n".join(filter(None, [private["player_1"], disc_summary, msg_map["player_1"]]))
            pending = {}

    # ── 非討論階段 ──
    else:
        player_inputs = {gs["players"][i]: ([user_input, *p_reply.values()][i]) for i in range(5)}
        ctrl_reply, ctrl_msgs, gs = run_controller(gs, player_inputs, ctrl_msgs)
        host_output, host_msgs = run_host(ctrl_reply, host_msgs)
        msg_map = _host_to_map(host_output)
        if gs["roles"]:
            private = build_private_messages(gs)
        if gs["current_phase"] in ("分派角色", "遊戲開始"):
            msg_map = {f"player_{i+1}": "" for i in range(5)}
            pending["player_1_private"] = private.get("player_1", "")
            player1_view = ""


        for k in (2, 3, 4, 5):
            base = "\n".join(filter(None, [private[f"player_{k}"], pending.get(f"player_{k}", ""), msg_map[f"player_{k}"]]))
            p_prompt[k] = base
            p_msg[k] = msg_map[f"player_{k}"]

        player1_view = "\n".join(filter(None, [
            pending.pop("player_1_private", ""),
            private.get("player_1", ""),
            pending.get("player_1", ""),
            msg_map["player_1"]
        ]))
        pending = {}

        # 若 player_1 沒有需要回應的訊息，自動推進直到需要人類行動
        while not msg_map.get("player_1", "").strip() and gs["winner"] is None:
            for k in (2, 3, 4, 5):
                if p_msg[k]:
                    p_reply[k], p_msgs[k] = call_ai(PLAYER_PROMPT, p_prompt[k], p_msgs[k])
                else:
                    p_reply[k] = ""
            player_inputs = {gs["players"][i]: (["", *p_reply.values()][i]) for i in range(5)}
            ctrl_reply, ctrl_msgs, gs = run_controller(gs, player_inputs, ctrl_msgs)
            host_output, host_msgs = run_host(ctrl_reply, host_msgs)
            msg_map = _host_to_map(host_output)

            if gs["roles"]:
                private = build_private_messages(gs)

            for k in (2, 3, 4, 5):
                p_prompt[k] = "\n".join(filter(None, [private[f"player_{k}"], msg_map[f"player_{k}"]]))
                p_msg[k] = msg_map[f"player_{k}"]

            player1_view = player1_view = "\n".join(filter(None, [
                pending.pop("player_1_private", ""),
                private.get("player_1", ""),
                pending.get("player_1", ""),
                msg_map["player_1"]
            ]))

            # 遊戲結束時顯示結果
            if gs["winner"]:
                player1_view = f"遊戲結束！{gs['winner']}方獲勝！\n" + player1_view
                break

    # ── 回寫 session state ──
    for k in (2, 3, 4, 5):
        ss[f"p{k}_msg"]      = p_msg[k]
        ss[f"p{k}_prompt"]   = p_prompt[k]
        ss[f"p{k}_messages"] = p_msgs[k]
    ss["controller_messages"] = ctrl_msgs
    ss["host_messages"]       = host_msgs
    ss["discussion_count"]    = disc_cnt
    ss["pending_discussion"]  = pending

    return player1_view, gs, ss

# ─────────────────────────────────────────────
# 10. Gradio 介面
# ─────────────────────────────────────────────
def interface_fn(user_input: str, gs: dict, ss: dict, history: list):
    reply, gs, ss = game_step(user_input, gs, ss)
    history = (history or []) + [
        {"role": "user",      "content": user_input},
        {"role": "assistant", "content": reply},
    ]
    return history, gs, ss, history

def reset_fn():
    gs, ss = initialize()
    return gs, ss, []

with gr.Blocks(title="AI 阿瓦隆") as demo:
    gr.Markdown("## AI 阿瓦隆 — 你是玩家1，與四位 AI 對戰")

    chatbot = gr.Chatbot(label="遊戲對話紀錄", height=500)
    user_input = gr.Textbox(label="玩家1 輸入", lines=2, placeholder="輸入你的行動或發言...")

    init_gs, init_ss = initialize()
    gs_state   = gr.State(value=init_gs)
    ss_state   = gr.State(value=init_ss)
    hist_state = gr.State(value=[])

    with gr.Row():
        submit_btn = gr.Button("送出", variant="primary")
        reset_btn  = gr.Button("重置遊戲")

    submit_btn.click(
        fn=interface_fn,
        inputs=[user_input, gs_state, ss_state, hist_state],
        outputs=[chatbot, gs_state, ss_state, hist_state],
    )
    reset_btn.click(
        fn=reset_fn,
        inputs=[],
        outputs=[gs_state, ss_state, chatbot]  # 加上 chatbot
    )

if __name__ == "__main__":
    demo.launch()
