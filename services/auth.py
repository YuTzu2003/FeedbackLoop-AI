import logging
from functools import wraps
from flask import Blueprint, request, session, redirect, url_for, render_template, flash, jsonify
from services.db import get_conn
import os, json, datetime

auth_bp = Blueprint('auth', __name__)

def login_log(user_id, ip, success, reason=""):
    log_dir = "tasks/cache"
    os.makedirs(log_dir, exist_ok=True) 

    log_entry = {
        "userid": user_id,
        "ip": ip,
        "login_time": datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        "success": success,
        "reason": reason
    }
    
    with open(f"{log_dir}/login_logs.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# ---- 登入驗證 ----
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask import current_app
        if current_app.config.get("TESTING") or os.environ.get("BYPASS_LOGIN"):
            if not session.get("id"):
                session["id"] = "test-id"
                session["userid"] = "test-user"
                session["name"] = "Test User"
                session["position"] = "Admin"
            return f(*args, **kwargs)
        if "id" not in session:
            if (request.path.startswith('/api/') or 
                request.path.startswith('/dashboard/upload') or 
                request.path.startswith('/dashboard/delete') or
                request.headers.get('X-Requested-With') == 'XMLHttpRequest' or
                'application/json' in request.headers.get('Accept', '')):
                return jsonify({"ok": False, "error": "請先登入系統"}), 401
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated_function

# ---- 管理員權限驗證 ----
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask import current_app
        if current_app.config.get("TESTING") or os.environ.get("BYPASS_LOGIN"):
            if not session.get("id"):
                session["id"] = "test-id"
                session["userid"] = "test-user"
                session["name"] = "Test User"
                session["position"] = "Admin"
            return f(*args, **kwargs)
        
        pos = session.get("position")
        if not pos or str(pos).strip().lower() != "admin":
            if (request.path.startswith('/api/') or 
                request.path.startswith('/dashboard/upload') or 
                request.path.startswith('/dashboard/delete') or
                request.headers.get('X-Requested-With') == 'XMLHttpRequest' or
                'application/json' in request.headers.get('Accept', '')):
                return jsonify({"ok": False, "error": "權限不足，需要管理員權限"}), 403
            flash("權限不足，無法存取此頁面", "danger")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated_function


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if "id" in session: 
        return redirect(url_for("index"))    
    if request.method == "POST":
        user_id = request.form["userid"]
        password = request.form["password"]
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT [ID], [UserID], [Password], [Name], [Position], [Location] FROM [dbo].[Users] WHERE UserID = ? AND Password = ?", (user_id, password))
        user = cursor.fetchone()
        
        if user:
            logging.info(f"使用者 {user_id} 登入成功")
            login_log(user_id, request.remote_addr, True, "登入成功")
            session["id"] = str(user.ID)
            session["userid"] = user.UserID
            session["name"] = user.Name
            session["position"] = user.Position
            session["location"] = user.Location
            cursor.execute("UPDATE [dbo].[Users] SET Last_login = GETDATE() WHERE ID = ?", (user.ID,))
            conn.commit()
            conn.close()
            return redirect("/")          
        conn.close()
        login_log(user_id, request.remote_addr, False, "帳號或密碼錯誤")
        return render_template("login.html", error="帳號或密碼錯誤")
    return render_template("login.html")

@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect("/")