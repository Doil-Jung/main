import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen

class ImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("이미지 준비 중...")
        self.setStyleSheet("background-color: #222; color: #fff;")
        self.setMouseTracking(True)
        
        # 투시 변환용 변수
        self.points = []  # 네 모서리 점들
        self.click_enabled = True
        
        # 숫자 영역 지정용 변수
        self.dragging = False
        self.start_pos = None
        self.end_pos = None
        self.digit_regions = []  # [(x, y, w, h), ...]
        
        # 현재 모드 (perspective 또는 digit_selection)
        self.mode = "perspective"
        
        # 돋보기
        self.magnifier = QLabel(self)
        self.magnifier.setFixedSize(80, 80)
        self.magnifier.setWindowFlags(Qt.ToolTip)
        self.magnifier.hide()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.mode == "perspective":
                self.handle_perspective_click(event.pos())
            elif self.mode == "digit_selection":
                self.handle_digit_drag_start(event.pos())

    def mouseMoveEvent(self, event):
        if self.mode == "perspective":
            self.handle_perspective_mouse_move(event.pos())
        elif self.mode == "digit_selection":
            self.handle_digit_drag_move(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.mode == "digit_selection":
            self.handle_digit_drag_end(event.pos())

    def leaveEvent(self, event):
        self.magnifier.hide()

    def handle_perspective_click(self, pos):
        """투시 변환용 네 점 클릭 처리"""
        if not self.click_enabled:
            return
            
        x, y = pos.x(), pos.y()
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
                
                # 소리 출력
                print('\a', end='', flush=True)
                
                if len(self.points) >= 4:
                    # 4개 점 모두 클릭 완료
                    self.parent().apply_perspective_transform()
                    QMessageBox.information(self, "완료", "네 점 클릭 완료! 투시 변환이 적용되었습니다.\n이제 숫자 영역을 지정할 수 있습니다.")
                else:
                    # 다음 점 안내
                    point_names = ["좌상단", "우상단", "우하단", "좌하단"]
                    print(f"다음 점 클릭: {point_names[len(self.points)]}")

    def handle_perspective_mouse_move(self, pos):
        """투시 변환 모드에서 마우스 이동 처리"""
        # 돋보기 기능
        if self.parent().current_image is not None:
            x, y = pos.x(), pos.y()
            # 실제 이미지 좌표로 변환
            label_w, label_h = self.width(), self.height()
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
                cv2.line(roi, (40, 0), (40, 79), (0, 255, 255), 1)
                cv2.line(roi, (0, 40), (79, 40), (0, 255, 255), 1)
                cv2.circle(roi, (40, 40), 4, (0, 255, 255), 1)
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
                self.magnifier.setPixmap(QPixmap.fromImage(qimg))
                # 돋보기 위치를 마우스 커서에 더 가깝게 조정
                self.magnifier.move(self.mapToGlobal(pos).x() + 10, self.mapToGlobal(pos).y() + 10)
                self.magnifier.show()

    def handle_digit_drag_start(self, pos):
        """숫자 영역 드래그 시작"""
        self.dragging = True
        self.start_pos = pos
        self.end_pos = pos

    def handle_digit_drag_move(self, pos):
        """숫자 영역 드래그 중"""
        if self.dragging:
            self.end_pos = pos
            self.update()
        
        # 돋보기 기능
        if self.parent().current_image is not None:
            x, y = pos.x(), pos.y()
            # 실제 이미지 좌표로 변환
            label_w, label_h = self.width(), self.height()
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
                cv2.line(roi, (40, 0), (40, 79), (0, 255, 255), 1)
                cv2.line(roi, (0, 40), (79, 40), (0, 255, 255), 1)
                cv2.circle(roi, (40, 40), 4, (0, 255, 255), 1)
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
                self.magnifier.setPixmap(QPixmap.fromImage(qimg))
                # 돋보기 위치를 마우스 커서에 더 가깝게 조정
                self.magnifier.move(self.mapToGlobal(pos).x() + 10, self.mapToGlobal(pos).y() + 10)
                self.magnifier.show()

    def handle_digit_drag_end(self, pos):
        """숫자 영역 드래그 종료"""
        if self.dragging:
            self.dragging = False
            self.end_pos = pos
            
            # 드래그 영역이 최소 크기 이상인지 확인
            if self.start_pos and self.end_pos:
                rect = QRect(self.start_pos, self.end_pos).normalized()
                if rect.width() > 10 and rect.height() > 10:
                    # 실제 이미지 좌표로 변환
                    label_w, label_h = self.width(), self.height()
                    img_h, img_w, _ = self.parent().current_image.shape
                    scale = min(label_w/img_w, label_h/img_h)
                    offset_x = (label_w - img_w*scale)/2
                    offset_y = (label_h - img_h*scale)/2
                    
                    x1 = int((rect.x() - offset_x) / scale)
                    y1 = int((rect.y() - offset_y) / scale)
                    x2 = int((rect.x() + rect.width() - offset_x) / scale)
                    y2 = int((rect.y() + rect.height() - offset_y) / scale)
                    
                    # 이미지 범위 내 좌표인지 확인
                    if (0 <= x1 < img_w and 0 <= y1 < img_h and 
                        0 <= x2 < img_w and 0 <= y2 < img_h):
                        x, y, w, h = min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)
                        self.digit_regions.append((x, y, w, h))
                        print(f"숫자 영역 {len(self.digit_regions)} 추가: ({x}, {y}) 크기 {w}x{h}")
                        
                        # 소리 출력
                        print('\a', end='', flush=True)
            
            self.start_pos = None
            self.end_pos = None
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.pixmap():
            painter = QPainter(self)
            
            if self.mode == "perspective":
                # 투시 변환 모드: 클릭된 점들 표시
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                for i, (x, y) in enumerate(self.points):
                    # 좌표 변환 (이미지 → 라벨)
                    label_w, label_h = self.width(), self.height()
                    img_h, img_w, _ = self.parent().current_image.shape
                    scale = min(label_w/img_w, label_h/img_h)
                    offset_x = (label_w - img_w*scale)/2
                    offset_y = (label_h - img_h*scale)/2
                    
                    label_x = int(x * scale + offset_x)
                    label_y = int(y * scale + offset_y)
                    
                    painter.setPen(QPen(Qt.green, 2))
                    painter.drawEllipse(label_x-3, label_y-3, 6, 6)
                    painter.drawText(label_x+8, label_y-8, str(i+1))
                
                # 네 점을 연결하는 사각형 그리기
                if len(self.points) >= 4:
                    corners = []
                    for x, y in self.points:
                        label_w, label_h = self.width(), self.height()
                        img_h, img_w, _ = self.parent().current_image.shape
                        scale = min(label_w/img_w, label_h/img_h)
                        offset_x = (label_w - img_w*scale)/2
                        offset_y = (label_h - img_h*scale)/2
                        label_x = int(x * scale + offset_x)
                        label_y = int(y * scale + offset_y)
                        corners.append((label_x, label_y))
                    
                    painter.setPen(QPen(Qt.green, 2))
                    for i in range(4):
                        painter.drawLine(corners[i][0], corners[i][1], 
                                       corners[(i+1)%4][0], corners[(i+1)%4][1])
            
            elif self.mode == "digit_selection":
                # 숫자 영역 지정 모드: 지정된 영역들 표시
                painter.setPen(QPen(Qt.red, 1, Qt.SolidLine))
                
                # 기존 숫자 영역들 그리기
                for i, (x, y, w, h) in enumerate(self.digit_regions):
                    # 좌표 변환 (이미지 → 라벨)
                    label_w, label_h = self.width(), self.height()
                    img_h, img_w, _ = self.parent().current_image.shape
                    scale = min(label_w/img_w, label_h/img_h)
                    offset_x = (label_w - img_w*scale)/2
                    offset_y = (label_h - img_h*scale)/2
                    
                    label_x = int(x * scale + offset_x)
                    label_y = int(y * scale + offset_y)
                    label_w_scaled = int(w * scale)
                    label_h_scaled = int(h * scale)
                    
                    painter.drawRect(label_x, label_y, label_w_scaled, label_h_scaled)
                    painter.drawText(label_x, label_y-5, f"{i+1}")
                
                # 현재 드래그 중인 영역 그리기
                if self.dragging and self.start_pos and self.end_pos:
                    rect = QRect(self.start_pos, self.end_pos).normalized()
                    painter.setPen(QPen(Qt.blue, 1, Qt.DashLine))
                    painter.drawRect(rect)

    def reset_points(self):
        """투시 변환 점들 재설정"""
        self.points = []

    def reset_regions(self):
        """숫자 영역 재설정"""
        self.digit_regions = []
        self.update()

    def set_mode(self, mode):
        """모드 변경"""
        self.mode = mode
        if mode == "digit_selection":
            self.click_enabled = False
        else:
            self.click_enabled = True

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("통합 숫자 인식 프로그램")
        self.resize(1200, 800)

        # 원본 이미지 로드
        self.original_image = cv2.imread('digit_thermo_cas.jpg')
        if self.original_image is None:
            print("digit_thermo_cas.jpg 파일을 로드할 수 없습니다.")
            self.original_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(self.original_image, "digit_thermo_cas.jpg not found", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 현재 표시할 이미지 (초기에는 원본)
        self.current_image = self.original_image.copy()
        
        # 투시 변환 결과 저장
        self.perspective_matrix = None
        self.corrected_image = None

        # UI 구성
        self.image_label = ImageWidget(self)
        self.image_timer = QTimer()
        self.image_timer.timeout.connect(self.update_image)
        self.image_timer.start(30)

        # 버튼들
        self.reset_perspective_btn = QPushButton("투시 변환 재설정")
        self.reset_perspective_btn.clicked.connect(self.reset_perspective)
        
        self.reset_digits_btn = QPushButton("숫자 영역 재설정")
        self.reset_digits_btn.clicked.connect(self.reset_digit_regions)
        
        self.save_btn = QPushButton("숫자 영역 저장")
        self.save_btn.clicked.connect(self.save_digit_regions)
        
        self.recognize_btn = QPushButton("숫자 인식")
        self.recognize_btn.clicked.connect(self.recognize_digits)
        self.recognize_btn.setEnabled(False)

        # 설명 텍스트
        self.guide = QTextEdit()
        self.guide.setReadOnly(True)
        self.guide.setPlainText(
            "사용 방법:\n"
            "1단계 - 투시 변환:\n"
            "  - 디스플레이의 네 모서리를 순서대로 클릭하세요:\n"
            "    * 1번: 좌상단\n"
            "    * 2번: 우상단\n"
            "    * 3번: 우하단\n"
            "    * 4번: 좌하단\n"
            "  - 네 점 클릭이 완료되면 자동으로 투시 변환이 적용됩니다.\n\n"
            "2단계 - 숫자 영역 지정:\n"
            "  - 투시 변환된 이미지에서 마우스 드래그로 숫자 영역을 지정하세요.\n"
            "  - 각 숫자마다 사각형으로 영역을 그려주세요.\n\n"
            "3단계 - 숫자 인식:\n"
            "  - 모든 숫자 영역을 지정한 후 '숫자 인식' 버튼을 누르세요.\n\n"
            "기타:\n"
            "  - '투시 변환 재설정': 1단계부터 다시 시작\n"
            "  - '숫자 영역 재설정': 2단계부터 다시 시작\n"
            "  - '숫자 영역 저장': 지정된 영역을 파일로 저장"
        )

        # 레이아웃
        grid = QGridLayout()
        grid.addWidget(self.image_label, 0, 0, 2, 2)  # 2x2 영역 차지

        right_panel = QVBoxLayout()
        right_panel.addWidget(self.reset_perspective_btn)
        right_panel.addWidget(self.reset_digits_btn)
        right_panel.addWidget(self.save_btn)
        right_panel.addWidget(self.recognize_btn)
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

    def update_image(self):
        """이미지 업데이트"""
        frame = self.current_image.copy()
        
        if self.image_label.mode == "perspective":
            # 투시 변환 모드: 클릭된 점들 표시
            if self.image_label.points:
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                for i, (x, y) in enumerate(self.image_label.points):
                    cv2.circle(frame, (int(x), int(y)), 3, colors[i], -1)
                    cv2.putText(frame, str(i+1), (int(x)+8, int(y)-8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
            
            # 네 점을 연결하는 사각형 그리기
            if len(self.image_label.points) >= 4:
                corners = np.array(self.image_label.points, dtype=np.int32)
                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
                cv2.putText(frame, "4-Point Rectangle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        elif self.image_label.mode == "digit_selection":
            # 숫자 영역 지정 모드: 아무것도 표시하지 않음
            pass
        
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
        roi = self.original_image[y_start:y_end, x_start:x_end]
        
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
            
            # 현재 이미지를 투시 변환된 이미지로 변경
            self.current_image = self.corrected_image.copy()
            
            # 모드를 숫자 영역 지정 모드로 변경
            self.image_label.set_mode("digit_selection")
            
            # 숫자 인식 버튼 활성화
            self.recognize_btn.setEnabled(True)
            
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

    def reset_perspective(self):
        """투시 변환 재설정"""
        self.image_label.reset_points()
        self.image_label.set_mode("perspective")
        self.current_image = self.original_image.copy()
        self.perspective_matrix = None
        self.corrected_image = None
        self.recognize_btn.setEnabled(False)
        print("투시 변환이 재설정되었습니다.")

    def reset_digit_regions(self):
        """숫자 영역 재설정"""
        self.image_label.reset_regions()
        self.recognize_btn.setEnabled(False)
        print("숫자 영역이 재설정되었습니다.")

    def save_digit_regions(self):
        """숫자 영역 저장"""
        if not self.image_label.digit_regions:
            QMessageBox.warning(self, "경고", "저장할 숫자 영역이 없습니다.")
            return
        
        # 각 숫자 영역을 개별 이미지로 저장
        for i, (x, y, w, h) in enumerate(self.image_label.digit_regions):
            digit_img = self.current_image[y:y+h, x:x+w]
            filename = f'integrated_digit_{i+1}.png'
            cv2.imwrite(filename, digit_img)
            print(f"{filename} 저장됨 (크기: {w}x{h})")
        
        # 영역 정보를 텍스트 파일로 저장
        with open('integrated_digit_regions.txt', 'w') as f:
            for i, (x, y, w, h) in enumerate(self.image_label.digit_regions):
                f.write(f"Digit {i+1}: ({x}, {y}) 크기 {w}x{h}\n")
        
        QMessageBox.information(self, "저장 완료", 
                               f"{len(self.image_label.digit_regions)}개 숫자 영역이 저장되었습니다.")

    def recognize_digits(self):
        """숫자 인식"""
        if not self.image_label.digit_regions:
            QMessageBox.warning(self, "경고", "인식할 숫자 영역이 없습니다.")
            return
        
        # 숫자 인식 로직
        recognized_digits = []
        
        for i, (x, y, w, h) in enumerate(self.image_label.digit_regions):
            digit_img = self.current_image[y:y+h, x:x+w]
            recognized_digit = self.recognize_single_digit(digit_img)
            recognized_digits.append(recognized_digit)
            print(f"숫자 {i+1}: 인식 결과 = {recognized_digit}")
        
        # 원본 이미지에 결과 표시
        self.display_results_on_original(recognized_digits)
        
        result_text = "인식 결과: " + " ".join(recognized_digits)
        QMessageBox.information(self, "숫자 인식 결과", result_text)

    def recognize_single_digit(self, digit_img):
        """단일 숫자 인식"""
        # 이미지 전처리
        gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 7세그먼트 영역 정의 (수정된 버전)
        h, w = cleaned.shape
        segments = {
            'a': (int(w*0.15), int(h*0), int(w*0.75), int(h*0.25)),  # 상단 가로
            'b': (int(w*0.6), int(h*0.1), int(w*0.4), int(h*0.4)),  # 우상단 세로
            'c': (int(w*0.6), int(h*0.6), int(w*0.4), int(h*0.4)),   # 우하단 세로
            'd': (int(w*0.15), int(h*0.75), int(w*0.75), int(h*0.25)),    # 하단 가로
            'e': (int(w*0.0), int(h*0.6), int(w*0.4), int(h*0.4)),    # 좌하단 세로
            'f': (int(w*0.0), int(h*0.1), int(w*0.4), int(h*0.4)),   # 좌상단 세로
            'g': (int(w*0.15), int(h*0.375), int(w*0.75), int(h*0.25))     # 중앙 가로
        }
        
        # 각 세그먼트 활성화 여부 확인
        segments_activation = {}
        for segment_name, region in segments.items():
            is_active, ratio = self.check_segment_activation(cleaned, region, threshold=0.2)  # 임계값을 0.1에서 0.15로 증가
            segments_activation[segment_name] = (is_active, ratio)
        
        # 숫자 인식
        recognized_digit = self.recognize_digit_pattern(segments_activation)
        
        # 디버깅 정보 출력
        active_segments = []
        for seg, activation_info in segments_activation.items():
            if isinstance(activation_info, tuple):
                is_active = activation_info[0]
                ratio = activation_info[1]
            else:
                is_active = activation_info
                ratio = 0
            if is_active:
                active_segments.append(seg)
            print(f"세그먼트 {seg}: 활성화={is_active}, 비율={ratio:.3f}")
        
        print(f"활성화된 세그먼트: {active_segments}")
        print(f"인식 결과: {recognized_digit}")
        
        return recognized_digit

    def check_segment_activation(self, img, segment_region, threshold=0.1):
        """세그먼트 활성화 여부 확인"""
        x, y, w, h = segment_region
        
        # 영역이 이미지 범위를 벗어나지 않도록 조정
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return False, 0
        
        # 해당 영역 추출
        roi = img[y:y+h, x:x+w]
        
        # 흰색 픽셀 비율 계산
        white_pixels = np.sum(roi == 255)
        total_pixels = roi.size
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
        
        return white_ratio > threshold, white_ratio

    def recognize_digit_pattern(self, segments_activation):
        """7세그먼트 패턴으로 숫자 인식"""
        # 7세그먼트 숫자 패턴 정의
        digit_patterns = {
            '0': ['a', 'b', 'c', 'd', 'e', 'f'],
            '1': ['b', 'c'],
            '2': ['a', 'b', 'd', 'e', 'g'],
            '3': ['a', 'b', 'c', 'd', 'g'],
            '4': ['b', 'c', 'f', 'g'],
            '5': ['a', 'c', 'd', 'f', 'g'],
            '6': ['a', 'c', 'd', 'e', 'f', 'g'],
            '7': ['a', 'b', 'c'],
            '8': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            '9': ['a', 'b', 'c', 'd', 'f', 'g']
        }
        
        # 활성화된 세그먼트 목록
        active_segments = []
        for seg, activation_info in segments_activation.items():
            if isinstance(activation_info, tuple):
                is_active = activation_info[0]
            else:
                is_active = activation_info
            if is_active:
                active_segments.append(seg)
        
        # 가장 유사한 패턴 찾기
        best_match = '?'
        best_score = -10
        
        for digit, pattern in digit_patterns.items():
            # 정확히 일치하는 세그먼트 수
            correct_segments = len(set(active_segments) & set(pattern))
            # 잘못 활성화된 세그먼트 수
            wrong_segments = len(set(active_segments) - set(pattern))
            # 누락된 세그먼트 수
            missing_segments = len(set(pattern) - set(active_segments))
            
            # 점수 계산 (정확한 세그먼트 - 잘못된 세그먼트*2 - 누락된 세그먼트)
            # 잘못된 세그먼트에 2배 페널티 부여
            score = correct_segments - (wrong_segments * 2) - missing_segments
            
            if score > best_score:
                best_score = score
                best_match = digit
        
        return best_match

    def display_results_on_original(self, recognized_digits):
        """원본 이미지에 인식 결과 표시"""
        # 원본 이미지에 투시 변환된 영역 표시
        result_img = self.original_image.copy()
        
        # 투시 변환된 영역을 원본 이미지에 표시
        if len(self.image_label.points) >= 4:
            corners = np.array(self.image_label.points, dtype=np.int32)
            cv2.polylines(result_img, [corners], True, (0, 255, 0), 2)
            
            # 투시 변환된 영역 내에서 숫자 위치 계산
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]
            x_start, x_end = int(np.min(x_coords)), int(np.max(x_coords))
            y_start, y_end = int(np.min(y_coords)), int(np.max(y_coords))
            
            # 각 숫자 영역을 원본 이미지 좌표로 변환
            for i, (x, y, w, h) in enumerate(self.image_label.digit_regions):
                # 투시 변환된 이미지에서의 상대 좌표를 원본 이미지의 절대 좌표로 변환
                # 투시 변환 행렬의 역행렬을 사용하여 정확한 좌표 변환
                if self.perspective_matrix is not None:
                    # 투시 변환된 이미지에서의 좌표
                    src_point = np.array([[[x + w/2, y + h/2]]], dtype=np.float32)
                    
                    # 역투시 변환으로 원본 이미지 좌표 계산
                    inv_matrix = cv2.invert(self.perspective_matrix)[1]
                    dst_point = cv2.perspectiveTransform(src_point, inv_matrix)
                    
                    original_x = int(dst_point[0][0][0] + x_start)
                    original_y = int(dst_point[0][0][1] + y_start)
                else:
                    # 투시 변환 행렬이 없는 경우 근사치 사용
                    original_x = x_start + x
                    original_y = y_start + y
                
                # 숫자만 표시 (사각형 제거)
                cv2.putText(result_img, recognized_digits[i], (original_x, original_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 결과 이미지 저장
        cv2.imwrite('final_result_with_digits.png', result_img)
        print("최종 결과 이미지 저장: final_result_with_digits.png")
        
        # 현재 이미지를 결과 이미지로 변경
        self.current_image = result_img.copy()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 한글 폰트 설정
    font = QFont("Malgun Gothic", 9)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 