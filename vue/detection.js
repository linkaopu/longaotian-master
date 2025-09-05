// O型圈检测系统前端逻辑

// 配置API基础URL
// 确保与后端服务器地址匹配
const API_BASE_URL = '/api';

// 登录状态管理
let currentUser = null;

// 检查登录状态
async function checkLoginStatus() {
    try {
        const response = await fetch('/api/check_login', {
            credentials: 'include'
        });
        const data = await response.json();

        if (data.logged_in) {
            currentUser = data.user;
            showUserInfo();
        } else {
            showLoginLink();
        }
    } catch (error) {
        console.error('检查登录状态失败:', error);
        showLoginLink();
    }
}

// 显示用户信息
function showUserInfo() {
    document.getElementById('user-info').classList.remove('hidden');
    document.getElementById('login-link').classList.add('hidden');
    document.getElementById('username-display').textContent = currentUser.username;

    // 用户登录后，自动更新统计信息
    fetchDailyStats();
}

// 显示登录链接
function showLoginLink() {
    document.getElementById('user-info').classList.add('hidden');
    document.getElementById('login-link').classList.remove('hidden');
    currentUser = null;
}

// 退出登录
async function logout() {
    try {
        const response = await fetch('/api/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include'
        });

        if (response.ok) {
            currentUser = null;
            showLoginLink();
            alert('已成功退出登录');
        } else {
            alert('退出登录失败，请重试');
        }
    } catch (error) {
        console.error('退出登录失败:', error);
        alert('退出登录失败，请重试');
    }
}

// 初始化函数
function initDetectionSystem() {
    // 检查登录状态
    checkLoginStatus();

    // 绑定退出登录按钮事件
    document.getElementById('logout-btn').addEventListener('click', logout);

    // 绑定处理按钮点击事件
    document.getElementById('process-btn').addEventListener('click', handleProcessClick);

    // 上传区域交互
    const uploadArea = document.getElementById('oring-upload-area');
    const imageUpload = document.getElementById('oring-image-upload');
    const uploadedFiles = document.getElementById('oring-uploaded-files');
    const originalImage = document.getElementById('original-image');
    const originalPlaceholder = document.getElementById('original-placeholder');

    uploadArea.addEventListener('click', () => {
        imageUpload.click();
    });

    imageUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            displayUploadedFiles(e.target.files);
            // 显示第一张图片作为预览
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (event) => {
                originalImage.src = event.target.result;
                originalImage.classList.remove('hidden');
                originalPlaceholder.classList.add('hidden');
            };
            reader.readAsDataURL(file);
        }
    });

    // 拖放功能
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('border-primary');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('border-primary');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('border-primary');

        if (e.dataTransfer.files.length > 0) {
            displayUploadedFiles(e.dataTransfer.files);
            // 显示第一张图片作为预览
            const file = e.dataTransfer.files[0];
            const reader = new FileReader();
            reader.onload = (event) => {
                originalImage.src = event.target.result;
                originalImage.classList.remove('hidden');
                originalPlaceholder.classList.add('hidden');
            };
            reader.readAsDataURL(file);
        }
    });

    // 显示上传的文件
    function displayUploadedFiles(files) {
        uploadedFiles.innerHTML = '';

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const fileSize = (file.size / (1024 * 1024)).toFixed(2);

            const fileItem = document.createElement('div');
            fileItem.className = 'flex items-center justify-between p-2 border border-gray-light rounded-lg mt-2';
            fileItem.innerHTML = `
          <div class="flex items-center">
            <i class="fa fa-file-image-o text-primary mr-2"></i>
            <div>
              <p class="text-sm truncate max-w-[200px]">${file.name}</p>
              <p class="text-xs text-gray-500">${fileSize} MB</p>
            </div>
          </div>
          <button class="text-gray-400 hover:text-danger transition-smooth">
            <i class="fa fa-times"></i>
          </button>
        `;

            // 删除文件功能
            fileItem.querySelector('button').addEventListener('click', () => {
                fileItem.remove();
                if (uploadedFiles.children.length === 0) {
                    originalImage.classList.add('hidden');
                    originalPlaceholder.classList.remove('hidden');
                }
            });

            uploadedFiles.appendChild(fileItem);
        }
    }

    // 如果用户已登录，拉取当日统计
    if (currentUser) {
        fetchDailyStats();
    }
}

// 处理按钮点击事件
function handleProcessClick() {
    // 检查登录状态
    if (!currentUser) {
        alert('请先登录后再进行检测');
        window.location.href = 'log.html?return=/';
        return;
    }

    const imageUpload = document.getElementById('oring-image-upload');
    const originalImage = document.getElementById('original-image');

    if (!originalImage.src || imageUpload.files.length === 0) {
        alert('请先上传图像');
        return;
    }

    // 显示处理中状态
    updateProcessingStatus(true);

    // 获取上传的文件
    const files = imageUpload.files;

    // 创建FormData对象
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    // 记录开始时间
    const startTime = Date.now();

    // 发送API请求到批量检测端点
    fetch(`${API_BASE_URL}/detect_batch`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
    })
        .then(response => {
            if (!response.ok) {
                if (response.status === 401) {
                    throw new Error('请先登录');
                }
                throw new Error('API请求失败');
            }
            return response.json();
        })
        .then(data => {
            // 计算处理时间
            const processTime = ((Date.now() - startTime) / 1000).toFixed(2);

            // 更新检测结果（显示第一个文件的结果）
            if (data.results && data.results.length > 0) {
                updateDetectionResults(data.results[0], processTime);

                // 添加所有文件到历史记录
                data.results.forEach(result => {
                    addToHistory(result.filename, result.passed);
                });

                // 更新统计数据
                updateStatistics(data.results);
                // 从后端刷新当日统计
                fetchDailyStats();
            }
        })
        .catch(error => {
            console.error('检测过程出错:', error);
            if (error.message === '请先登录') {
                alert('请先登录后再进行检测');
                window.location.href = 'log.html?return=/';
            } else {
                alert('检测过程出错: ' + error.message);
            }
            updateProcessingStatus(false);
        });
}

// 更新处理状态
function updateProcessingStatus(isProcessing) {
    const processedPlaceholder = document.getElementById('processed-placeholder');
    const processedImage = document.getElementById('processed-image');
    const detectionResult = document.getElementById('detection-result');

    if (isProcessing) {
        processedPlaceholder.textContent = '处理中...';
        processedPlaceholder.classList.remove('hidden');
        processedImage.classList.add('hidden');
        detectionResult.className = 'inline-block px-4 py-2 rounded-full text-white font-medium bg-gray-medium';
        detectionResult.textContent = '等待检测';
    } else {
        processedPlaceholder.classList.add('hidden');
        processedImage.classList.remove('hidden');
    }
}

// 更新检测结果
function updateDetectionResults(data, processTime) {
    // 更新处理后图像
    const processedImage = document.getElementById('processed-image');
    processedImage.src = 'data:image/png;base64,' + data.processed_image;
    processedImage.classList.remove('hidden');
    document.getElementById('processed-placeholder').classList.add('hidden');

    // 用校正后的图像覆盖原始显示（符合“先标定再检测”的主流程）
    const originalImage = document.getElementById('original-image');
    const originalPlaceholder = document.getElementById('original-placeholder');
    if (data.calibrated_image) {
        originalImage.src = 'data:image/png;base64,' + data.calibrated_image;
        originalImage.classList.remove('hidden');
        if (originalPlaceholder) originalPlaceholder.classList.add('hidden');
    }

    // 如果有带圆轮廓的原图，则显示它
    const originalWithCirclesImage = document.getElementById('original-with-circles-image');
    if (data.original_with_circles && originalWithCirclesImage) {
        originalWithCirclesImage.src = 'data:image/png;base64,' + data.original_with_circles;
        originalWithCirclesImage.classList.remove('hidden');
        document.getElementById('original-with-circles-placeholder').classList.add('hidden');
    }

    // 更新检测结果状态
    const detectionResult = document.getElementById('detection-result');
    if (data.passed) {
        detectionResult.className = 'inline-block px-4 py-2 rounded-full text-white font-medium bg-success';
        detectionResult.textContent = '检测通过';
    } else {
        detectionResult.className = 'inline-block px-4 py-2 rounded-full text-white font-medium bg-danger';
        detectionResult.textContent = '检测失败';
    }

    // 更新检测详情
    document.getElementById('centroid-coords').textContent = data.centroid ? `${data.centroid[0]}, ${data.centroid[1]}` : '-,-';
    document.getElementById('inner-radius').textContent = data.radius && data.radius.length > 0 ? `${data.radius[0]} px` : '- px';
    document.getElementById('outer-radius').textContent = data.radius && data.radius.length > 1 ? `${data.radius[1]} px` : '- px';
    document.getElementById('process-time').textContent = `${data.process_time || processTime} s`;

    // 更新直方图（如果有数据）
    if (window.histogramChart && data.histogram) {
        window.histogramChart.data.datasets[0].data = data.histogram;
        window.histogramChart.update();
    }
}

// 更新统计数据
function updateStatistics(results) {
    // 计算通过和失败的数量
    const passedCount = results.filter(result => result.passed).length;
    const failedCount = results.filter(result => !result.passed).length;

    // 更新通过/失败数量
    document.getElementById('passed-count').textContent = passedCount;
    document.getElementById('failed-count').textContent = failedCount;

    // 计算总检测数量
    const totalCount = passedCount + failedCount;

    // 如果有总数量显示元素，也更新它
    const totalElement = document.getElementById('total-count');
    if (totalElement) {
        totalElement.textContent = totalCount;
    }

    // 计算平均处理时间（假设我们有每个结果的处理时间）
    // 实际应用中，可能需要从API响应中获取每个结果的处理时间
    // document.getElementById('avg-time').textContent = `${avgTime}s`;
}

// 从后端读取当日统计
async function fetchDailyStats() {
    try {
        const resp = await fetch(`${API_BASE_URL}/stats/daily`, {
            credentials: 'include'
        });
        if (!resp.ok) return;
        const data = await resp.json();
        if (typeof data.passed === 'number' && typeof data.failed === 'number') {
            document.getElementById('passed-count').textContent = data.passed;
            document.getElementById('failed-count').textContent = data.failed;
            document.getElementById('total-count').textContent = data.total;

            // 更新平均处理时间
            if (data.avg_time) {
                document.getElementById('avg-time').textContent = `${data.avg_time}s`;
            }
        }
    } catch (e) {
        // 忽略错误，保持本地统计
        console.error('获取每日统计失败:', e);
    }
}

// 添加到历史记录
function addToHistory(filename, passed) {
    const historyRecords = document.getElementById('history-records');

    // 如果是"暂无记录"，先清空
    if (historyRecords.querySelector('tr td[colspan="4"]')) {
        historyRecords.innerHTML = '';
    }

    const resultText = passed ? '通过' : '失败';
    const resultClass = passed ? 'text-success' : 'text-danger';
    const resultIcon = passed ? 'fa-check-circle' : 'fa-times-circle';

    const now = new Date();
    const timeString = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;

    const record = document.createElement('tr');
    record.className = 'hover:bg-gray-50 transition-smooth';
    record.innerHTML = `
        <td class="py-3 text-sm">${filename}</td>
        <td class="py-3 text-sm">
          <span class="${resultClass} flex items-center">
            <i class="fa ${resultIcon} mr-1"></i> ${resultText}
          </span>
        </td>
        <td class="py-3 text-sm text-gray-500">${timeString}</td>
        <td class="py-3 text-sm">
          <button class="text-primary hover:underline mr-3">查看</button>
          <button class="text-gray-500 hover:text-gray-700">删除</button>
        </td>
      `;

    // 添加删除功能
    record.querySelector('button:last-child').addEventListener('click', () => {
        record.remove();
        if (historyRecords.children.length === 0) {
            historyRecords.innerHTML = `
            <tr>
              <td colspan="4" class="py-4 text-center text-gray-500 text-sm">暂无记录</td>
            </tr>
          `;
        }
    });

    historyRecords.prepend(record);
}

// 当页面加载完成后初始化系统
window.addEventListener('DOMContentLoaded', initDetectionSystem);