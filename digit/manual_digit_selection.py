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
        
        # 드래그 관련 변수
        self.dragging = False
        self.start_pos = None
        self.end_pos = None
        self.digit_regions = []  # [(x, y, w, h), ...]
        
        # 돋보기
        self.magnifier = QLabel(self)
        self.magnifier.setFixedSize(80, 80)
        self.magnifier.setWindowFlags(Qt.ToolTip)
        self.magnifier.hide()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.start_pos = event.pos()
            self.end_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.end_pos = event.pos()
            self.update()  # 위젯 다시 그리기
        
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
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.end_pos = event.pos()
            
            # 드래그 영역이 최소 크기 이상인지 확인
            if self.start_pos and self.end_pos:
                rect = QRect(self.start_pos, self.end_pos).normalized()
                if rect.width() > 10 and rect.height() > 10:
                    # 실제 이미지 좌표로 변환
                    label_w, label_h = self.parent().image_label.width(), self.parent().image_label.height()
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

    def leaveEvent(self, event):
        self.magnifier.hide()

    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.pixmap():
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            
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
                painter.setPen(QPen(Qt.blue, 2, Qt.DashLine))
                painter.drawRect(rect)

    def reset_regions(self):
        self.digit_regions = []
        self.update()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("숫자 영역 수동 지정 프로그램")
        self.resize(1200, 800)

        # 원본 이미지 로드
        self.original_image = cv2.imread('digit_thermo_cas.jpg')
        if self.original_image is None:
            print("digit_thermo_cas.jpg 파일을 로드할 수 없습니다.")
            self.original_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(self.original_image, "digit_thermo_cas.jpg not found", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 투시 변환된 이미지 로드
        self.current_image = cv2.imread('perspective_corrected.png')
        if self.current_image is None:
            print("perspective_corrected.png 파일을 로드할 수 없습니다.")
            self.current_image = self.original_image.copy()
        
        # UI 구성
        self.image_label = ImageWidget(self)
        self.image_timer = QTimer()
        self.image_timer.timeout.connect(self.update_image)
        self.image_timer.start(30)

        # 버튼들
        self.reset_btn = QPushButton("영역 재설정")
        self.reset_btn.clicked.connect(self.reset_regions)
        
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
            "1. 투시 변환된 이미지에서 마우스 드래그로 숫자 영역을 지정하세요.\n"
            "2. 각 숫자마다 사각형으로 영역을 그려주세요.\n"
            "3. 모든 숫자 영역을 지정한 후 '숫자 인식' 버튼을 누르세요.\n"
            "4. '영역 재설정' 버튼으로 처음부터 다시 시작할 수 있습니다.\n"
            "5. '숫자 영역 저장' 버튼으로 지정된 영역을 파일로 저장할 수 있습니다."
        )

        # 레이아웃
        grid = QGridLayout()
        grid.addWidget(self.image_label, 0, 0, 2, 2)  # 2x2 영역 차지

        right_panel = QVBoxLayout()
        right_panel.addWidget(self.reset_btn)
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
        
        # 지정된 숫자 영역 표시
        for i, (x, y, w, h) in enumerate(self.image_label.digit_regions):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def reset_regions(self):
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
            filename = f'manual_digit_{i+1}.png'
            cv2.imwrite(filename, digit_img)
            print(f"{filename} 저장됨 (크기: {w}x{h})")
        
        # 영역 정보를 텍스트 파일로 저장
        with open('digit_regions.txt', 'w') as f:
            for i, (x, y, w, h) in enumerate(self.image_label.digit_regions):
                f.write(f"Digit {i+1}: ({x}, {y}) 크기 {w}x{h}\n")
        
        QMessageBox.information(self, "저장 완료", 
                               f"{len(self.image_label.digit_regions)}개 숫자 영역이 저장되었습니다.")

    def recognize_digits(self):
        """숫자 인식"""
        if not self.image_label.digit_regions:
            QMessageBox.warning(self, "경고", "인식할 숫자 영역이 없습니다.")
            return
        
        # 숫자 인식 로직 (기존 템플릿 매칭 방식 사용)
        recognized_digits = []
        
        for i, (x, y, w, h) in enumerate(self.image_label.digit_regions):
            digit_img = self.current_image[y:y+h, x:x+w]
            # 여기에 숫자 인식 로직 추가
            # 임시로 "?" 표시
            recognized_digits.append("?")
            print(f"숫자 {i+1}: 인식 결과 = ?")
        
        result_text = "인식 결과: " + " ".join(recognized_digits)
        QMessageBox.information(self, "숫자 인식 결과", result_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 한글 폰트 설정
    font = QFont("Malgun Gothic", 9)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 