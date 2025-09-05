import sys
import os
from functools import wraps
from flask import Blueprint, request, jsonify, session, send_file
import cv2 as cv
import numpy as np
import base64
import io
from PIL import Image
import json
import hashlib
import time
import datetime

# 导入B00107064.py中的核心功能
from src.B00107064 import (
    threshold as threshold_bin,
    image_hist,
    component_label,
    remove_smallest_areas,
    get_centroid,
    make_bounding_box,
    calculate_radius,
    oring_result,
    draw_faulty_locations,
    draw_bounding_box,
    closing,
    paint_labels
)

# 导入cv.py中的相机标定功能
from src.cv import (
    mtx,
    dist,
)

# DB utils
from db import insert_detection, daily_stats, list_failures, get_failure_image

# 创建蓝图
detection_bp = Blueprint('detection', __name__)

# 形态学结构用于图像清理
morph_struct = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]

# 辅助函数: 将BGR图像转换为RGB
def bgr_to_rgb(img_bgr):
    if img_bgr is None:
        return None
    if len(img_bgr.shape) == 2:
        return cv.cvtColor(img_bgr, cv.COLOR_GRAY2RGB)
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

# 辅助函数: 将图像转换为base64编码
def to_base64_png(img):
    if img is None:
        return ""
    if img.ndim == 2:
        pil_img = Image.fromarray(img)
    else:
        pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# 检测O型圈的核心函数
def detect_oring(image_bgr):
    try:
        start_time = time.time()
        # 第一步：相机标定与去畸变（若可用）
        calibrated_bgr = image_bgr
        try:
            if mtx is not None and dist is not None and image_bgr is not None:
                h, w = image_bgr.shape[:2]
                newMatrix, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                calibrated_bgr = cv.undistort(image_bgr, mtx, dist, None, newMatrix)
        except Exception:
            # 如果去畸变失败，继续使用原图
            calibrated_bgr = image_bgr

        # 可选：保存标定后的图像到与B00107064.py一致的目录结构，便于对照排查
        try:
            save_dir = os.path.join(project_root, 'src', 'images', 'calibrated', '4')
            os.makedirs(save_dir, exist_ok=True)
            filename = f"upload_{int(time.time()*1000)}.png"
            cv.imwrite(os.path.join(save_dir, filename), calibrated_bgr)
        except Exception:
            pass
        
        # 处理图像：阈值
        image_threshold = threshold_bin(calibrated_bgr)
        if image_threshold is None:
            return None, "无效的图像用于阈值处理"

        # 添加形态学处理来清理图像噪声
        # 使用B00107064.py中的closing函数
        struct = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        image_threshold = closing(image_threshold, struct)

        # 形态学 + 连通区域标记
        image_labels = component_label(image_threshold)
        image_labels = remove_smallest_areas(image_labels)

        # 计算质心和边界框
        centroid = get_centroid(image_labels)
        bbox = make_bounding_box(image_labels)

        # 与B00107064.py主流程一致：先将label涂成二值环（白环/黑底）
        drawn_image = paint_labels(image_threshold.copy(), image_labels)
        # 转换为RGB以便显示
        drawn_image_rgb = bgr_to_rgb(drawn_image)
        radius = calculate_radius(drawn_image_rgb, centroid)
        result = oring_result(drawn_image_rgb, centroid, radius)

        # 绘制结果
        if result[0] is False:
            drawn_image_rgb = draw_faulty_locations(drawn_image_rgb, result[2])
        drawn_image_rgb = draw_bounding_box(drawn_image_rgb, bbox, result[0])

        # 在原图上绘制所有圆的轮廓和圆形，并在左上角显示检测结果和圆心坐标
        original_with_circles = bgr_to_rgb(calibrated_bgr.copy())
        
        # 绘制内外圆
        if centroid and len(centroid) >= 2:
            # 绘制内圆
            if len(radius) >= 1:
                cv.circle(original_with_circles, (int(centroid[1]), int(centroid[0])), int(radius[0]), (0, 255, 0), 2)
            # 绘制外圆
            if len(radius) >= 2:
                cv.circle(original_with_circles, (int(centroid[1]), int(centroid[0])), int(radius[1]), (0, 255, 0), 2)
        
        # 在左上角显示检测结果
        result_text = "PASSED" if result[0] else "FAILED"
        color = (0, 255, 0) if result[0] else (0, 0, 255)
        cv.putText(original_with_circles, result_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 显示圆心坐标
        if centroid and len(centroid) >= 2:
            centroid_text = f"Centroid: ({int(centroid[1])}, {int(centroid[0])})"
            cv.putText(original_with_circles, centroid_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 显示半径信息
        if len(radius) >= 2:
            radius_text = f"Radius: inner={int(radius[0])}, outer={int(radius[1])}"
            cv.putText(original_with_circles, radius_text, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 准备返回结果
        original_rgb = bgr_to_rgb(calibrated_bgr)
        # 直方图（基于灰度的标定图像）
        try:
            gray_for_hist = cv.cvtColor(calibrated_bgr, cv.COLOR_BGR2GRAY)
            histogram = image_hist(gray_for_hist).tolist()
        except Exception:
            histogram = None

        processing_time = round(time.time() - start_time, 3)
        result_data = {
            "passed": bool(result[0]),
            "faulty_coords": result[1],
            "centroid": centroid,
            "radius": radius,
            "bbox": bbox,
            "original_image": to_base64_png(original_rgb),
            "calibrated_image": to_base64_png(original_rgb),
            "processed_image": to_base64_png(drawn_image_rgb),
            "original_with_circles": to_base64_png(original_with_circles),  # 新增带圆轮廓的原图
            "histogram": histogram,
            "process_time": processing_time
        }

        return result_data, None
    except Exception as e:
        return None, str(e)

# 登录验证装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 添加调试信息
        print(f"Session data: {session}")
        print(f"Session logged_in: {session.get('logged_in')}")
        if not session.get('logged_in'):
            return jsonify({'error': '请先登录'}), 401
        return f(*args, **kwargs)
    return decorated_function

# API路由: 检测O型圈
@detection_bp.route('/detect', methods=['POST'])
@login_required
def api_detect():
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '没有文件上传'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        # 读取图像文件
        image_stream = io.BytesIO(file.read())
        image = Image.open(image_stream)
        image_bgr = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

        # 调用检测函数
        result_data, error = detect_oring(image_bgr)

        if error:
            return jsonify({'error': error}), 400

        if result_data is None:
            return jsonify({'error': '检测失败'}), 500

        # 记录数据库
        try:
            insert_detection(
                filename=file.filename,
                passed=bool(result_data.get('passed')),
                detected_at=datetime.datetime.now(),
                centroid=result_data.get('centroid'),
                radius=result_data.get('radius'),
                bbox=result_data.get('bbox'),
                histogram=result_data.get('histogram'),
                process_time_seconds=result_data.get('process_time'),
                username=session.get('username'),
                original_image_b64=result_data.get('original_image'),
                processed_image_b64=result_data.get('processed_image'),
                overlay_image_b64=result_data.get('original_with_circles'),
            )
        except Exception as db_exc:
            # 不影响主流程
            pass

        return jsonify(result_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API路由: 批量检测O型圈
@detection_bp.route('/detect_batch', methods=['POST'])
@login_required
def api_detect_batch():
    print('Received request to /detect_batch')
    try:
        # 检查是否有文件上传
        if 'files' not in request.files:
            return jsonify({'error': '没有文件上传'}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': '没有选择文件'}), 400

        results = []
        for file in files:
            if file.filename == '':
                continue

            # 读取图像文件
            image_stream = io.BytesIO(file.read())
            image = Image.open(image_stream)
            image_bgr = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

            # 调用检测函数
            result_data, error = detect_oring(image_bgr)

            if error:
                results.append({'filename': file.filename, 'error': error})
            elif result_data is None:
                results.append({'filename': file.filename, 'error': '检测失败'})
            else:
                result_data['filename'] = file.filename
                results.append(result_data)
                # 记录数据库
                try:
                    insert_detection(
                        filename=file.filename,
                        passed=bool(result_data.get('passed')),
                        detected_at=datetime.datetime.now(),
                        centroid=result_data.get('centroid'),
                        radius=result_data.get('radius'),
                        bbox=result_data.get('bbox'),
                        histogram=result_data.get('histogram'),
                        process_time_seconds=result_data.get('process_time'),
                        username=session.get('username'),
                        original_image_b64=result_data.get('original_image'),
                        processed_image_b64=result_data.get('processed_image'),
                        overlay_image_b64=result_data.get('original_with_circles'),
                    )
                except Exception:
                    pass

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API路由: 健康检查
@detection_bp.route('/health', methods=['GET'])
def api_health():
    return jsonify({"status": "ok"})


# 统计：某日检测数量
@detection_bp.route('/stats/daily', methods=['GET'])
def api_stats_daily():
    date_ymd = request.args.get('date')  # YYYY-MM-DD
    try:
        stats = daily_stats(date_ymd)
        return jsonify({
            'date': date_ymd or datetime.date.today().strftime('%Y-%m-%d'),
            'total': stats['total'],
            'passed': stats['passed'],
            'failed': stats['failed'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 获取不合格记录列表（含缩略图）
@detection_bp.route('/failures', methods=['GET'])
def api_failures():
    date_ymd = request.args.get('date')
    try:
        limit = int(request.args.get('limit', '50'))
        offset = int(request.args.get('offset', '0'))
    except Exception:
        limit, offset = 50, 0
    try:
        rows = list_failures(date_ymd=date_ymd, limit=limit, offset=offset)
        return jsonify({'date': date_ymd, 'items': rows})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 获取某条不合格记录的图片
@detection_bp.route('/failure/<int:detect_id>/image', methods=['GET'])
def api_failure_image(detect_id: int):
    kind = request.args.get('kind', 'overlay')
    raw = request.args.get('raw') in ('1', 'true', 'yes')
    try:
        img_b64 = get_failure_image(detect_id, kind)
        if not img_b64:
            return jsonify({'error': '未找到图片'}), 404
        if raw:
            import base64, io
            data = base64.b64decode(img_b64)
            return send_file(io.BytesIO(data), mimetype='image/png')
        return jsonify({'id': detect_id, 'kind': kind, 'image': img_b64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Import user management functions from DB
from db import create_user, get_user_by_username, get_user_by_email, update_last_login

# 辅助函数: 哈希密码
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# API路由: 用户登录
@detection_bp.route('/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': '邮箱和密码不能为空'}), 400

        # 从MySQL数据库获取用户信息（通过邮箱）
        user = get_user_by_email(email)
        if not user:
            return jsonify({'error': '邮箱或密码错误'}), 401

        # 检查密码
        if user['password_hash'] != hash_password(password):
            return jsonify({'error': '邮箱或密码错误'}), 401

        # 检查用户是否激活
        if not user.get('is_active', True):
            return jsonify({'error': '账户已被禁用'}), 403

        # 更新最后登录时间
        update_last_login(user['username'])

        # 设置会话
        session['logged_in'] = True
        session['username'] = user['username']
        session['department'] = user['department']
        session['user_id'] = user['id']

        return jsonify({'message': '登录成功', 'user': {'username': user['username'], 'department': user['department']}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API路由: 用户注册
@detection_bp.route('/register', methods=['POST'])
def api_register():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        if not username or not email or not password:
            return jsonify({'error': '所有字段都是必填的'}), 400

        # 验证用户名长度
        if len(username) < 4 or len(username) > 20:
            return jsonify({'error': '用户名长度必须为4-20个字符'}), 400

        # 验证邮箱格式
        if not '@' in email:
            return jsonify({'error': '邮箱格式不正确'}), 400

        # 验证密码长度
        if len(password) < 8:
            return jsonify({'error': '密码长度至少为8位'}), 400

        # 验证密码包含字母和数字
        if not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
            return jsonify({'error': '密码必须包含字母和数字'}), 400

        # 检查用户名是否已存在
        existing_user = get_user_by_username(username)
        if existing_user:
            return jsonify({'error': '用户名已存在'}), 400

        # 检查邮箱是否已存在
        existing_email = get_user_by_email(email)
        if existing_email:
            return jsonify({'error': '邮箱已被注册'}), 400

        # 保存新用户到MySQL数据库（设置默认部门为'user'）
        success = create_user(username, email, hash_password(password), 'user')
        if not success:
            return jsonify({'error': '注册失败，请稍后重试'}), 500

        return jsonify({'message': '注册成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API路由: 检查登录状态
@detection_bp.route('/check_login', methods=['GET'])
def api_check_login():
    if session.get('logged_in'):
        return jsonify({
            'logged_in': True,
            'user': {
                'username': session.get('username'),
                'department': session.get('department')
            }
        })
    else:
        return jsonify({'logged_in': False})

# API路由: 用户登出
@detection_bp.route('/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'message': '登出成功'})


# API路由: 创建管理员用户（仅用于初始化）
@detection_bp.route('/create-admin', methods=['POST'])
def api_create_admin():
    try:
        data = request.get_json()
        username = data.get('username', 'admin')
        password = data.get('password', 'admin123456')
        email = data.get('email', 'admin@example.com')
        department = data.get('department', 'management')
        
        # 检查是否已存在用户
        existing_user = get_user_by_username(username)
        if existing_user:
            return jsonify({'error': '用户已存在'}), 400
            
        # 创建管理员用户
        success = create_user(username, email, hash_password(password), department)
        if success:
            return jsonify({'message': f'管理员用户 {username} 创建成功'})
        else:
            return jsonify({'error': '创建失败'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API路由: 获取每日检测统计
@detection_bp.route('/stats/daily', methods=['GET'])
@login_required
def api_daily_stats():
    try:
        date_str = request.args.get('date')
        from db import get_daily_detection_stats
        stats = get_daily_detection_stats(date_str)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API路由: 获取周统计
@detection_bp.route('/stats/weekly', methods=['GET'])
@login_required
def api_weekly_stats():
    try:
        days = request.args.get('days', 7, type=int)
        from db import get_weekly_stats
        stats = get_weekly_stats(days)
        return jsonify({'stats': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API路由: 获取检测历史
@detection_bp.route('/stats/history', methods=['GET'])
@login_required
def api_detection_history():
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        from db import list_detections
        detections = list_detections(limit, offset, date_from, date_to)
        return jsonify({'detections': detections})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API路由: 获取检测历史统计
@detection_bp.route('/stats/history_summary', methods=['GET'])
@login_required
def api_history_summary():
    try:
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        from db import get_history_summary
        summary = get_history_summary(date_from, date_to)
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500