import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_digit_image():
    """숫자 이미지 로드"""
    img = cv2.imread('integrated_digit_1.png')
    if img is None:
        print("integrated_digit_2.png 파일을 찾을 수 없습니다.")
        return None
    return img

def preprocess_digit_image(img):
    """숫자 이미지 전처리"""
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 이진화
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return gray, binary, cleaned

def define_seven_segment_regions(img):
    """7세그먼트 영역 정의"""
    h, w = img.shape
    
    # 7세그먼트 위치 정의 (상대적 비율)
    segments = {
        'a': (int(w*0.15), int(h*0), int(w*0.75), int(h*0.15)),  # 상단 가로
        'b': (int(w*0.6), int(h*0.1), int(w*0.4), int(h*0.3)),  # 우상단 세로
        'c': (int(w*0.6), int(h*0.6), int(w*0.4), int(h*0.35)),   # 우하단 세로
        'd': (int(w*0.15), int(h*0.85), int(w*0.75), int(h*0.15)),    # 하단 가로
        'e': (int(w*0.0), int(h*0.6), int(w*0.4), int(h*0.3)),    # 좌하단 세로
        'f': (int(w*0.0), int(h*0.1), int(w*0.4), int(h*0.3)),   # 좌상단 세로
        'g': (int(w*0.15), int(h*0.4), int(w*0.75), int(h*0.2))     # 중앙 가로
    }
    
    return segments

def check_segment_activation(img, segment_region, threshold=0.1):
    """세그먼트 활성화 여부 확인"""
    x, y, w, h = segment_region
    
    # 영역이 이미지 범위를 벗어나지 않도록 조정
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return False
    
    # 해당 영역 추출
    roi = img[y:y+h, x:x+w]
    
    # 흰색 픽셀 비율 계산
    white_pixels = np.sum(roi == 255)
    total_pixels = roi.size
    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
    
    return white_ratio > threshold, white_ratio

def recognize_digit(segments_activation):
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
    
    # 활성화된 세그먼트 목록 (수정: 튜플에서 첫 번째 값만 확인)
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
    best_score = 0
    
    for digit, pattern in digit_patterns.items():
        # 정확히 일치하는 세그먼트 수
        correct_segments = len(set(active_segments) & set(pattern))
        # 잘못 활성화된 세그먼트 수
        wrong_segments = len(set(active_segments) - set(pattern))
        # 누락된 세그먼트 수
        missing_segments = len(set(pattern) - set(active_segments))
        
        # 점수 계산 (정확한 세그먼트 - 잘못된 세그먼트 - 누락된 세그먼트)
        score = correct_segments - wrong_segments - missing_segments
        
        if score > best_score:
            best_score = score
            best_match = digit
    
    return best_match, active_segments

def visualize_segments(img, segments, segments_activation):
    """세그먼트 시각화"""
    result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 7개 세그먼트를 무지개 색상으로 정의 (BGR 순서)
    segment_colors = {
        'a': (0, 0, 255),    # 빨강
        'b': (0, 165, 255),  # 주황
        'c': (0, 255, 255),  # 노랑
        'd': (0, 255, 0),    # 초록
        'e': (255, 0, 0),    # 파랑
        'f': (255, 0, 255),  # 남색
        'g': (255, 0, 255)   # 보라
    }
    
    for segment_name, (x, y, w, h) in segments.items():
        # 각 세그먼트별로 고유한 색상 사용
        color = segment_colors[segment_name]
        
        # 사각형 그리기 (선 두께를 1로 설정)
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 1)
    
    return result_img

def main():
    # 숫자 이미지 로드
    img = load_digit_image()
    if img is None:
        return
    
    print(f"이미지 크기: {img.shape}")
    
    # 이미지 전처리
    gray, binary, cleaned = preprocess_digit_image(img)
    
    # 전처리 결과 저장
    cv2.imwrite('digit_test_gray.png', gray)
    cv2.imwrite('digit_test_binary.png', binary)
    cv2.imwrite('digit_test_cleaned.png', cleaned)
    
    # 7세그먼트 영역 정의
    segments = define_seven_segment_regions(cleaned)
    
    # 각 세그먼트 활성화 여부 확인
    segments_activation = {}
    for segment_name, region in segments.items():
        is_active, ratio = check_segment_activation(cleaned, region, threshold=0.1)
        segments_activation[segment_name] = (is_active, ratio)
        print(f"세그먼트 {segment_name}: 활성화={is_active}, 비율={ratio:.3f}")
    
    # 숫자 인식
    recognized_digit, active_segments = recognize_digit(segments_activation)
    print(f"\n인식 결과: {recognized_digit}")
    print(f"활성화된 세그먼트: {active_segments}")
    
    # 결과 시각화
    result_img = visualize_segments(cleaned, segments, segments_activation)
    
    # 결과 이미지에 인식된 숫자 표시
    cv2.putText(result_img, f"Recognized: {recognized_digit}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imwrite('digit_recognition_result.png', result_img)
    print("결과 이미지 저장: digit_recognition_result.png")
    
    # matplotlib로 결과 표시 (선택사항)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(cleaned, cmap='gray')
        axes[1, 0].set_title('Preprocessed Binary')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Recognition Result: {recognized_digit}')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('digit_recognition_analysis.png', dpi=150, bbox_inches='tight')
        print("분석 이미지 저장: digit_recognition_analysis.png")
        plt.show()
        
    except ImportError:
        print("matplotlib이 설치되지 않아 그래프를 표시할 수 없습니다.")

if __name__ == "__main__":
    main() 