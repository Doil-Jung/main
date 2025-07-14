import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QComboBox, QFileDialog, QTextEdit, QMessageBox, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import datetime

class WebcamWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.points = []
        self.temp_values = []
        self.setText("웹캠 준비 중...")
        self.setStyleSheet("background-color: #222; color: #fff;")
        self.setMouseTracking(True)
        self.click_enabled = True
        self.magnifier = QLabel(self)
        self.magnifier.setFixedSize(80, 80)
        self.magnifier.setWindowFlags(Qt.ToolTip)
        self.magnifier.hide()

    def mousePressEvent(self, event):
        if self.click_enabled and event.button() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            self.points.append((x, y))
            self.parent().on_webcam_click(x, y)

    def mouseMoveEvent(self, event):
        if self.parent().current_frame is not None:
            x = event.pos().x()
            y = event.pos().y()
            # 실제 프레임 좌표로 변환 (on_webcam_click과 동일)
            label_w, label_h = self.parent().webcam_label.width(), self.parent().webcam_label.height()
            frame_h, frame_w, _ = self.parent().current_frame.shape
            scale = min(label_w/frame_w, label_h/frame_h)
            offset_x = (label_w - frame_w*scale)/2
            offset_y = (label_h - frame_h*scale)/2
            fx = int((x - offset_x) / scale)
            fy = int((y - offset_y) / scale)
            # 20x20 영역 추출
            x0, y0 = max(0, fx-10), max(0, fy-10)
            x1, y1 = min(frame_w, fx+10), min(frame_h, fy+10)
            roi = self.parent().current_frame[y0:y1, x0:x1]
            if roi.size > 0:
                roi = cv2.resize(roi, (80, 80), interpolation=cv2.INTER_NEAREST)
                # 중앙에 십자선 그리기
                cv2.line(roi, (40, 0), (40, 79), (0, 255, 255), 1)  # 세로선
                cv2.line(roi, (0, 40), (79, 40), (0, 255, 255), 1)  # 가로선
                # 중앙에 작은 원도 추가(선택)
                cv2.circle(roi, (40, 40), 4, (0, 255, 255), 1)
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
                self.magnifier.setPixmap(QPixmap.fromImage(qimg))
                self.magnifier.move(event.globalX()+20, event.globalY()+20)
                self.magnifier.show()

    def leaveEvent(self, event):
        self.magnifier.hide()

    def reset_points(self):
        self.points = []
        self.temp_values = []

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.set_xlabel("시간")
        self.ax.set_ylabel("온도 (°C)")
        self.fig.tight_layout()

    def plot(self, times, temps):
        self.ax.clear()
        self.ax.plot(times, temps, marker='o')
        self.ax.set_xlabel("시간")
        self.ax.set_ylabel("온도 (°C)")
        self.ax.grid(True)
        self.fig.autofmt_xdate()
        self.draw()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("온도계 자동 기록 프로그램")
        self.resize(1200, 800)

        # 데이터
        self.data = pd.DataFrame(columns=["시간", "온도"])
        self.sampling_rate = 1  # seconds
        self.timer = QTimer()
        self.timer.timeout.connect(self.measure_temperature)
        self.is_measuring = False

        # 웹캠 대신 이미지 파일 사용
        self.cap = None
        self.test_image = cv2.imread('thermostat_slope.jpg')
        if self.test_image is None:
            print("thermostat_slope.jpg 파일을 로드할 수 없습니다.")
            self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(self.test_image, "thermostat_slope.jpg not found", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        self.webcam_label = WebcamWidget(self)
        self.webcam_timer = QTimer()
        self.webcam_timer.timeout.connect(self.update_webcam)
        self.webcam_timer.start(30)

        # 그래프
        self.canvas = MatplotlibCanvas(self)

        # 표
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["시간", "온도(°C)"])

        # 좌하단: 설정/버튼/설명
        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(["1초", "5초", "10초", "30초", "1분"])
        self.sampling_combo.currentIndexChanged.connect(self.change_sampling_rate)

        self.start_btn = QPushButton("측정 시작")
        self.start_btn.clicked.connect(self.start_measurement)
        self.stop_btn = QPushButton("측정 종료")
        self.stop_btn.clicked.connect(self.stop_measurement)
        self.stop_btn.setEnabled(False)

        self.export_data_btn = QPushButton("데이터 내보내기")
        self.export_data_btn.clicked.connect(self.export_data)
        self.export_graph_btn = QPushButton("그래프 저장")
        self.export_graph_btn.clicked.connect(self.export_graph)

        self.guide = QTextEdit()
        self.guide.setReadOnly(True)
        self.guide.setPlainText(
            "사용 방법:\n"
            "1. 웹캠 화면에서 기준이 되는 눈금 2곳을 차례로 클릭하고, 각각의 온도값을 입력하세요.\n"
            "2. 샘플링 주기를 선택하고 '측정 시작'을 누르면 자동 기록이 시작됩니다.\n"
            "3. 데이터와 그래프는 각각 파일로 저장할 수 있습니다.\n"
            "4. 측정 종료 후 웹캠 화면을 클릭하면 기준점 재설정이 가능합니다."
        )

        # 레이아웃
        grid = QGridLayout()
        grid.addWidget(self.webcam_label, 0, 0, 2, 2)  # 2x2 영역 차지
        grid.addWidget(self.canvas, 0, 2)
        grid.addWidget(self.table, 1, 2)

        left_bottom = QVBoxLayout()
        left_bottom.addWidget(QLabel("샘플링 주기"))
        left_bottom.addWidget(self.sampling_combo)
        left_bottom.addWidget(self.start_btn)
        left_bottom.addWidget(self.stop_btn)
        left_bottom.addWidget(self.export_data_btn)
        left_bottom.addWidget(self.export_graph_btn)
        left_bottom.addWidget(QLabel("사용 설명"))
        left_bottom.addWidget(self.guide)
        left_bottom.addStretch()
        grid.addLayout(left_bottom, 2, 0, 1, 3)

        grid.setColumnStretch(0, 2)  # 웹캠 부분 크게
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 1)  # 나머지 작게
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 1)

        self.setLayout(grid)

        # 온도계 기준점
        self.ref_points = []
        self.ref_temps = []
        self.red_color = None

    def update_webcam(self):
        # 웹캠 대신 이미지 파일 사용
        frame = self.test_image.copy()
        self.current_frame = frame.copy()
        # 기준점 표시
        for idx, pt in enumerate(self.ref_points):
            cv2.circle(frame, pt, 8, (0, 255, 0), -1)
            cv2.putText(frame, f"{self.ref_temps[idx]}°C", (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        # 빨간 기둥 끝점 자동 인식
        red_tip = None
        temp = None
        if len(self.ref_points) == 2:
            red_tip = self.find_column_tip_in_roi(frame)
            if red_tip:
                y1, y2 = self.ref_points[0][1], self.ref_points[1][1]
                t1, t2 = self.ref_temps[0], self.ref_temps[1]
                y_red = red_tip[1]
                if y1 != y2:
                    temp = (t2 - t1) / (y2 - y1) * (y_red - y1) + t1
                cv2.circle(frame, red_tip, 8, (0, 0, 255), -1)
                if temp is not None:
                    cv2.putText(frame, f"{temp:.2f}°C", (red_tip[0]+10, red_tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.webcam_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.webcam_label.width(), self.webcam_label.height(), Qt.KeepAspectRatio))

    def on_webcam_click(self, x, y):
        # 실제 프레임 좌표로 변환
        label_w, label_h = self.webcam_label.width(), self.webcam_label.height()
        frame_h, frame_w, _ = self.current_frame.shape
        scale = min(label_w/frame_w, label_h/frame_h)
        offset_x = (label_w - frame_w*scale)/2
        offset_y = (label_h - frame_h*scale)/2
        fx = int((x - offset_x) / scale)
        fy = int((y - offset_y) / scale)
        if len(self.ref_points) < 2:
            temp, ok = self.get_temp_input(len(self.ref_points)+1)
            if ok:
                self.ref_points.append((fx, fy))
                self.ref_temps.append(temp)
        elif self.red_color is None:
            self.red_color = self.current_frame[fy, fx]
            QMessageBox.information(self, "설정 완료", "기준점과 빨간 기둥 색상 설정이 완료되었습니다.\n이제 측정 시작이 가능합니다.")
        else:
            # 기준점 재설정
            self.ref_points = []
            self.ref_temps = []
            self.red_color = None
            self.data = pd.DataFrame(columns=["시간", "온도"])
            self.update_table()
            self.canvas.plot([], [])
            QMessageBox.information(self, "재설정", "기준점 두 곳과 빨간 기둥 색상을 다시 지정하세요.")

    def get_temp_input(self, idx):
        from PyQt5.QtWidgets import QInputDialog
        temp, ok = QInputDialog.getDouble(self, "눈금 온도 입력", f"{idx}번째 클릭한 눈금의 온도(°C)를 입력하세요:", 0, -100, 200, 1)
        return temp, ok

    def change_sampling_rate(self, idx):
        rates = [1, 5, 10, 30, 60]
        self.sampling_rate = rates[idx]

    def start_measurement(self):
        if len(self.ref_points) < 2:
            QMessageBox.warning(self, "설정 필요", "기준점 두 곳을 먼저 지정하세요.")
            return
        # 측정 시작 시 ROI 이진화 이미지 한 번만 저장
        if self.current_frame is not None:
            self.save_roi_bin_debug(self.current_frame)
        self.is_measuring = True
        self.timer.start(self.sampling_rate * 1000)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_measurement(self):
        self.is_measuring = False
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def measure_temperature(self):
        # 빨간 기둥 끝점의 y좌표로 온도 계산 (자동 인식)
        if len(self.ref_points) < 2:
            return
        red_tip = self.find_column_tip_in_roi(self.current_frame)
        if not red_tip:
            return
        y1, y2 = self.ref_points[0][1], self.ref_points[1][1]
        t1, t2 = self.ref_temps[0], self.ref_temps[1]
        y_red = red_tip[1]
        if y1 == y2:
            return
        temp = (t2 - t1) / (y2 - y1) * (y_red - y1) + t1
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.data = pd.concat([self.data, pd.DataFrame([[now, temp]], columns=["시간", "온도"])] , ignore_index=True)
        self.update_table()
        self.update_graph()

    def update_table(self):
        self.table.setRowCount(len(self.data))
        for i, row in self.data.iterrows():
            self.table.setItem(i, 0, QTableWidgetItem(str(row["시간"])))
            self.table.setItem(i, 1, QTableWidgetItem(f"{row['온도']:.2f}"))

    def update_graph(self):
        times = self.data["시간"].tolist()
        temps = self.data["온도"].tolist()
        self.canvas.plot(times, temps)

    def export_data(self):
        path, _ = QFileDialog.getSaveFileName(self, "데이터 내보내기", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if path:
            if path.endswith('.csv'):
                self.data.to_csv(path, index=False, encoding='utf-8-sig')
            elif path.endswith('.xlsx'):
                self.data.to_excel(path, index=False)

    def export_graph(self):
        path, _ = QFileDialog.getSaveFileName(self, "그래프 저장", "", "PNG Files (*.png);;JPG Files (*.jpg)")
        if path:
            self.canvas.fig.savefig(path)

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

    def find_red_tip(self, frame):
        # HSV 색공간 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 빨간색 범위 마스크 (예시, 필요시 조정)
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 100])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        # 윤곽선 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 가장 큰 빨간 영역
            c = max(contours, key=cv2.contourArea)
            # y좌표가 가장 작은 점(온도계 위쪽이 0)
            tip = tuple(c[c[:,:,1].argmin()][0])
            return tip
        return None

    def find_column_tip_in_roi(self, frame, width=10, thresh_val=180):
        if len(self.ref_points) < 2:
            return None
        pt1, pt2 = self.ref_points
        
        # ROI 영역 계산
        roi_x_start = max(0, int(min(pt1[0], pt2[0]) - width//2))
        roi_x_end = min(frame.shape[1], int(max(pt1[0], pt2[0]) + width//2))
        y_start = max(0, int(min(pt1[1], pt2[1])))
        y_end = min(frame.shape[0], int(max(pt1[1], pt2[1])))
        
        # ROI 영역 잘라내기
        roi = frame[y_start:y_end, roi_x_start:roi_x_end]
        
        # 이진화 (save_roi_bin_debug와 동일한 방법)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_bin = cv2.threshold(roi_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # 아무런 처리 없이 가장 큰 연결된 흰색 영역 찾기
        contours, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug_img = cv2.cvtColor(roi_bin, cv2.COLOR_GRAY2BGR)
        
        if not contours:
            cv2.imwrite('roi_bin_debug_contour.png', debug_img)
            return None
            
        # 가장 큰 contour 찾기 (면적 기준)
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        
        print(f"가장 큰 contour 면적: {largest_area:.1f}")
        
        # 면적이 너무 작으면 무시
        if largest_area < 100:
            cv2.imwrite('roi_bin_debug_contour.png', debug_img)
            return None
        
        # 모든 contour 그리기(녹색)
        cv2.drawContours(debug_img, contours, -1, (0,255,0), 1)
        # 가장 큰 contour(파란색)
        cv2.drawContours(debug_img, [largest_contour], -1, (255,0,0), 2)
        
        # 가장 큰 contour의 가장 위쪽 점 찾기
        topmost = tuple(largest_contour[largest_contour[:,:,1].argmin()][0])
        
        # ROI 영역 좌표를 원본 이미지 좌표로 변환
        x_tip = roi_x_start + topmost[0]
        y_tip = y_start + topmost[1]
        
        # 기둥 끝점(빨간 원)
        cv2.circle(debug_img, topmost, 4, (0,0,255), -1)
        
        # 디버그 정보 추가
        cv2.putText(debug_img, f"Area: {largest_area:.0f}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(debug_img, f"Top: {topmost}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        cv2.imwrite('roi_bin_debug_contour.png', debug_img)
        return (x_tip, y_tip)

    def save_roi_bin_debug(self, frame, width=10, thresh_val=180):
        if len(self.ref_points) < 2:
            return
        pt1, pt2 = self.ref_points
        
        # ROI 영역 계산
        roi_x_start = max(0, int(min(pt1[0], pt2[0]) - width//2))
        roi_x_end = min(frame.shape[1], int(max(pt1[0], pt2[0]) + width//2))
        y_start = max(0, int(min(pt1[1], pt2[1])))
        y_end = min(frame.shape[0], int(max(pt1[1], pt2[1])))
        
        # ROI 영역만 잘라내기
        roi = frame[y_start:y_end, roi_x_start:roi_x_end]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_bin = cv2.threshold(roi_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite('roi_bin_debug.png', roi_bin)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())