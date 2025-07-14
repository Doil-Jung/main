import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

class ImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("이미지 준비 중...")
        self.setStyleSheet("background-color: #222; color: #fff;")
        self.setMouseTracking(True)
        self.click_enabled = True
        self.magnifier = QLabel(self)
        self.magnifier.setFixedSize(80, 80)
        self.magnifier.setWindowFlags(Qt.ToolTip)
        self.magnifier.hide()
        
        # 클릭된 점들을 저장
        self.points = []

    def mousePressEvent(self, event):
        if self.click_enabled and event.button() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            self.handle_click(x, y)

    def handle_click(self, x, y):
        """클릭 처리 - 네 모서리 설정"""
        # 실제 이미지 좌표로 변환
        label_w, label_h = self.width(), self.height()
        if self.parent().current_image is not None:
            img_h, img_w, _ = self.parent().current_image.shape
            scale = min(label_w/img_w, label_h/img_h)
            offset_x = (label_w - img_w*scale)/2
            offset_y = (label_h - img_h*scale)/2
            fx = int((x - offset_x) / scale)
            fy = int((y - offset_y) / scale)
            
            # 이미지 범위 내 좌표인지 확인
            if 0 <= fx < img_w and 0 <= fy < img_h:
                self.points.append((fx, fy))
                
                print(f"점 {len(self.points)} 클릭: ({fx}, {fy})")
                
                # 소리 출력 (콘솔에서 벨 문자)
                print('\a', end='', flush=True)
                
                if len(self.points) >= 4:
                    # 4개 점 모두 클릭 완료
                    self.parent().apply_perspective_transform()
                    QMessageBox.information(self, "완료", "네 점 클릭 완료! 투시 변환이 적용되었습니다.")
                else:
                    # 다음 점 안내
                    point_names = ["좌상단", "우상단", "우하단", "좌하단"]
                    print(f"다음 점 클릭: {point_names[len(self.points)]}")

    def mouseMoveEvent(self, event):
        # 돋보기 기능
        if self.parent().current_image is not None:
            x = event.pos().x()
            y = event.pos().y()
            # 실제 이미지 좌표로 변환
            label_w, label_h = self.parent().image_label.width(), self.parent().image_label.height()
            img_h, img_w, _ = self.parent().current_image.shape
            scale = min(label_w/img_w, label_h/img_h)
            offset_x = (label_w - img_w*scale)/2
            offset_y = (label_h - img_h*scale)/2
            fx = int((x - offset_x) / scale)
            fy = int((y - offset_y) / scale)
            # 20x20 영역 추출
            x0, y0 = max(0, fx-10), max(0, fy-10)
            x1, y1 = min(img_w, fx+10), min(img_h, fy+10)
            roi = self.parent().current_image[y0:y1, x0:x1]
            if roi.size > 0:
                roi = cv2.resize(roi, (80, 80), interpolation=cv2.INTER_NEAREST)
                # 중앙에 십자선 그리기
                cv2.line(roi, (40, 0), (40, 79), (0, 255, 255), 1)  # 세로선
                cv2.line(roi, (0, 40), (79, 40), (0, 255, 255), 1)  # 가로선
                # 중앙에 작은 원도 추가
                cv2.circle(roi, (40, 40), 4, (0, 255, 255), 1)
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
                self.magnifier.setPixmap(QPixmap.fromImage(qimg))
                self.magnifier.move(event.globalX()+20, event.globalY()+20)
                self.magnifier.show()

    def mouseReleaseEvent(self, event):
        pass

    def leaveEvent(self, event):
        self.magnifier.hide()

    def reset_points(self):
        self.points = []

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("투시 변환 테스트 프로그램")
        self.resize(1200, 800)

        # 이미지 로드
        self.current_image = cv2.imread('digit_thermo_cas.jpg')
        if self.current_image is None:
            print("digit_thermo_cas.jpg 파일을 로드할 수 없습니다.")
            self.current_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(self.current_image, "digit_thermo_cas.jpg not found", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # UI 구성
        self.image_label = ImageWidget(self)
        self.image_timer = QTimer()
        self.image_timer.timeout.connect(self.update_image)
        self.image_timer.start(30)

        # 버튼들
        self.reset_btn = QPushButton("재설정")
        self.reset_btn.clicked.connect(self.reset_points)
        
        self.save_btn = QPushButton("결과 저장")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)

        # 설명 텍스트
        self.guide = QTextEdit()
        self.guide.setReadOnly(True)
        self.guide.setPlainText(
            "사용 방법:\n"
            "1. 디스플레이의 네 모서리를 순서대로 클릭하세요:\n"
            "   - 1번: 좌상단\n"
            "   - 2번: 우상단\n" 
            "   - 3번: 우하단\n"
            "   - 4번: 좌하단\n"
            "2. 네 점 클릭이 완료되면 자동으로 투시 변환이 적용됩니다.\n"
            "3. '결과 저장' 버튼을 눌러 변환 결과를 파일로 저장할 수 있습니다.\n"
            "4. '재설정' 버튼을 눌러 처음부터 다시 시작할 수 있습니다."
        )

        # 레이아웃
        grid = QGridLayout()
        grid.addWidget(self.image_label, 0, 0, 2, 2)  # 2x2 영역 차지

        right_panel = QVBoxLayout()
        right_panel.addWidget(self.reset_btn)
        right_panel.addWidget(self.save_btn)
        right_panel.addWidget(QLabel("사용 설명"))
        right_panel.addWidget(self.guide)
        right_panel.addStretch()
        grid.addLayout(right_panel, 0, 2, 2, 1)

        grid.setColumnStretch(0, 2)  # 이미지 부분 크게
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 1)  # 나머지 작게
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        self.setLayout(grid)

        # 투시 변환 결과 저장
        self.perspective_matrix = None
        self.corrected_image = None

    def update_image(self):
        """이미지 업데이트"""
        frame = self.current_image.copy()
        
        # 클릭된 점 표시
        if self.image_label.points:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            for i, (x, y) in enumerate(self.image_label.points):
                cv2.circle(frame, (int(x), int(y)), 5, colors[i], -1)
                cv2.putText(frame, str(i+1), (int(x)+10, int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        
        # 네 점을 연결하는 사각형 그리기
        if len(self.image_label.points) >= 4:
            corners = np.array(self.image_label.points, dtype=np.int32)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            cv2.putText(frame, "4-Point Rectangle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # 네 점을 포함하는 최소 직사각형도 표시 (투시 변환용 ROI)
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]
            x_start, x_end = int(np.min(x_coords)), int(np.max(x_coords))
            y_start, y_end = int(np.min(y_coords)), int(np.max(y_coords))
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 255), 1)
            cv2.putText(frame, "Perspective ROI", (x_start, y_start-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def apply_perspective_transform(self):
        """투시 변환 적용"""
        if len(self.image_label.points) != 4:
            print("4개 점이 클릭되지 않았습니다.")
            return
        
        # 클릭된 네 점 좌표
        corners = np.array(self.image_label.points, dtype=np.float32)
        
        # 네 점을 꼭짓점으로 하는 사각형의 경계 계산
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]
        x_start, x_end = int(np.min(x_coords)), int(np.max(x_coords))
        y_start, y_end = int(np.min(y_coords)), int(np.max(y_coords))
        
        print(f"ROI 영역: ({x_start}, {y_start}) ~ ({x_end}, {y_end})")
        
        # ROI 영역 추출 (네 점을 포함하는 최소 직사각형)
        roi = self.current_image[y_start:y_end, x_start:x_end]
        
        # ROI 내에서의 상대 좌표로 변환
        src_points = corners - np.array([x_start, y_start])
        
        # 네 점의 경계를 계산하여 목표 사각형 크기 결정
        src_x_coords = src_points[:, 0]
        src_y_coords = src_points[:, 1]
        target_w = int(np.max(src_x_coords) - np.min(src_x_coords))
        target_h = int(np.max(src_y_coords) - np.min(src_y_coords))
        
        # 네 점을 꼭짓점으로 하는 정면 사각형 정의
        dst_points = np.array([
            [0, 0],                    # 좌상단
            [target_w, 0],             # 우상단
            [target_w, target_h],      # 우하단
            [0, target_h]              # 좌하단
        ], dtype=np.float32)
        
        print(f"src_points: {src_points}")
        print(f"dst_points: {dst_points}")
        print(f"목표 사각형 크기: {target_w} x {target_h}")
        
        # 투시 변환 행렬 계산
        try:
            # OpenCV가 요구하는 형태로 변환 (4x1x2 형태)
            src_points_4x1x2 = src_points.reshape(4, 1, 2)
            dst_points_4x1x2 = dst_points.reshape(4, 1, 2)
            
            print(f"투시 변환 전 src_points_4x1x2 형태: {src_points_4x1x2.shape}")
            print(f"투시 변환 전 dst_points_4x1x2 형태: {dst_points_4x1x2.shape}")
            
            # OpenCV가 요구하는 정확한 형태로 변환
            src_points_opencv = np.float32(src_points_4x1x2)
            dst_points_opencv = np.float32(dst_points_4x1x2)
            
            self.perspective_matrix = cv2.getPerspectiveTransform(src_points_opencv, dst_points_opencv)
            
            # 투시 변환 적용
            self.corrected_image = cv2.warpPerspective(roi, self.perspective_matrix, (target_w, target_h))
            
            # 디버그용 저장
            cv2.imwrite('perspective_corrected.png', self.corrected_image)
            print("투시 변환 완료 - perspective_corrected.png 저장됨")
            
            # 투시 변환 결과를 시각화한 이미지도 저장
            self.save_perspective_debug_image(roi, src_points, dst_points)
            
            # UI 업데이트
            self.save_btn.setEnabled(True)
            
        except cv2.error as e:
            print(f"투시 변환 오류: {e}")
            QMessageBox.warning(self, "오류", f"투시 변환 중 오류가 발생했습니다: {e}")

    def save_perspective_debug_image(self, original_roi, src_points, dst_points):
        """투시 변환 디버그 이미지 저장"""
        # 원본 ROI에 클릭된 점 표시
        debug_img = original_roi.copy()
        
        # 클릭된 점들을 원으로 표시
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # 녹, 파, 빨, 청록
        for i, (x, y) in enumerate(src_points):
            cv2.circle(debug_img, (int(x), int(y)), 5, colors[i], -1)
            cv2.putText(debug_img, str(i+1), (int(x)+10, int(y)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        
        # 점들을 선으로 연결
        for i in range(4):
            pt1 = tuple(map(int, src_points[i]))
            pt2 = tuple(map(int, src_points[(i+1)%4]))
            cv2.line(debug_img, pt1, pt2, (255, 255, 255), 2)
        
        cv2.imwrite('perspective_debug.png', debug_img)
        print("투시 변환 디버그 이미지 저장 - perspective_debug.png")

    def save_results(self):
        """결과 저장"""
        if self.corrected_image is not None:
            cv2.imwrite('perspective_corrected.png', self.corrected_image)
            QMessageBox.information(self, "저장 완료", 
                                   "투시 변환 결과가 저장되었습니다:\n"
                                   "- perspective_corrected.png: 투시 변환된 결과\n"
                                   "- perspective_debug.png: 원본 ROI와 클릭된 점들")

    def reset_points(self):
        """점들 재설정"""
        self.image_label.reset_points()
        self.perspective_matrix = None
        self.corrected_image = None
        self.save_btn.setEnabled(False)
        print("점들이 재설정되었습니다.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 한글 폰트 설정
    font = QFont("Malgun Gothic", 9)  # Windows 기본 한글 폰트
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 