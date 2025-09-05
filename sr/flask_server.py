import os
from flask import Flask, render_template, send_from_directory, session
from flask_cors import CORS

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算Vue文件夹的绝对路径
vue_dir = os.path.abspath(os.path.join(current_dir, '../vue'))

from oring_detection_api import detection_bp
from db import init_db

app = Flask(__name__, static_folder=vue_dir, template_folder=vue_dir)
# 设置密钥用于会话
app.secret_key = 'your-secret-key-change-in-production'

# 添加CORS支持，允许携带credentials
# 配置CORS以支持凭证和正确的来源
CORS(app, supports_credentials=True, origins=["http://localhost:5000", "http://127.0.0.1:5000"])

# 设置会话配置
app.config['SESSION_COOKIE_SECURE'] = False  # 在开发环境中设置为False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

app.register_blueprint(detection_bp, url_prefix='/api')

@app.route('/<path:path>')
def serve_static(path):
    # 确保请求的文件存在于vue目录中
    requested_file = os.path.join(vue_dir, path)
    if os.path.exists(requested_file) and os.path.isfile(requested_file):
        return send_from_directory(vue_dir, path)
    else:
        # 如果文件不存在，返回404错误
        return 'File not found', 404


@app.route('/')
def index():
    return send_from_directory(vue_dir, 'a.html')

@app.route('/login')
def login():
    return send_from_directory(vue_dir, 'log.html')

@app.route('/register')
def register():
    return send_from_directory(vue_dir, 'register.html')

@app.route('/api/test')
def api_test():
    return {'status': 'success', 'message': 'Flask API is working'}

if __name__ == '__main__':
    if not os.path.exists(vue_dir):
        print(f'Error: Vue folder not found at {vue_dir}')
    else:
        try:
            init_db()
            print('MySQL tables ensured.')
        except Exception as e:
            print(f'Warning: Failed to initialize MySQL tables: {e}')
        print('Starting Flask server on http://127.0.0.1:5000')
        app.run(debug=True, host='0.0.0.0', port=5000)