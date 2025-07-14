import cv2
import numpy as np

def load_digit_image():
    """숫자 이미지 로드"""
    img = cv2.imread('integrated_digit_1.png')
    if img is None:
        print("integrated_digit_1.png 파일을 찾을 수 없습니다.")
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
        'a': (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.15)),  # 상단 가로
        'b': (int(w*0.85), int(h*0.15), int(w*0.15), int(h*0.35)),  # 우상단 세로
        'c': (int(w*0.85), int(h*0.5), int(w*0.15), int(h*0.35)),   # 우하단 세로
        'd': (int(w*0.1), int(h*0.75), int(w*0.8), int(h*0.15)),    # 하단 가로
        'e': (int(w*0.0), int(h*0.5), int(w*0.15), int(h*0.35)),    # 좌하단 세로
        'f': (int(w*0.0), int(h*0.15), int(w*0.15), int(h*0.35)),   # 좌상단 세로
        'g': (int(w*0.1), int(h*0.45), int(w*0.8), int(h*0.15))     # 중앙 가로
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
        return False, 0
    
    # 해당 영역 추출
    roi = img[y:y+h, x:x+w]
    
    # 흰색 픽셀 비율 계산
    white_pixels = np.sum(roi == 255)
    total_pixels = roi.size
    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
    
    return white_ratio > threshold, white_ratio

def recognize_digit_detailed(segments_activation):
    """7세그먼트 패턴으로 숫자 인식 (상세 분석)"""
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
    for seg, (is_active, ratio) in segments_activation.items():
        if is_active:
            active_segments.append(seg)
    
    print(f"활성화된 세그먼트: {active_segments}")
    
    # 각 숫자 패턴과 비교
    results = {}
    for digit, pattern in digit_patterns.items():
        # 정확히 일치하는 세그먼트 수
        correct_segments = len(set(active_segments) & set(pattern))
        # 잘못 활성화된 세그먼트 수
        wrong_segments = len(set(active_segments) - set(pattern))
        # 누락된 세그먼트 수
        missing_segments = len(set(pattern) - set(active_segments))
        
        # 점수 계산
        score = correct_segments - wrong_segments - missing_segments
        
        results[digit] = {
            'pattern': pattern,
            'correct': correct_segments,
            'wrong': wrong_segments,
            'missing': missing_segments,
            'score': score
        }
        
        print(f"숫자 {digit}: 정확={correct_segments}, 잘못={wrong_segments}, 누락={missing_segments}, 점수={score}")
    
    # 최고 점수 찾기
    best_match = max(results.keys(), key=lambda x: results[x]['score'])
    best_score = results[best_match]['score']
    
    return best_match, best_score, results

def visualize_segments_detailed(img, segments, segments_activation):
    """세그먼트 시각화 (상세)"""
    result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for segment_name, (x, y, w, h) in segments.items():
        is_active, ratio = segments_activation[segment_name]
        
        # 활성화 여부에 따라 색상 결정
        if is_active:
            color = (0, 255, 0)  # 녹색 (활성화)
        else:
            color = (0, 0, 255)  # 빨간색 (비활성화)
        
        # 사각형 그리기
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        
        # 세그먼트 이름 표시
        cv2.putText(result_img, segment_name, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 활성화 비율 표시
        cv2.putText(result_img, f"{ratio:.2f}", (x, y+h+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return result_img

def save_individual_segments(img, segments, segments_activation):
    """각 세그먼트를 개별 이미지로 저장"""
    for segment_name, (x, y, w, h) in segments.items():
        # 영역이 이미지 범위를 벗어나지 않도록 조정
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        
        if w > 0 and h > 0:
            roi = img[y:y+h, x:x+w]
            is_active, ratio = segments_activation[segment_name]
            filename = f'segment_{segment_name}_active_{is_active}_ratio_{ratio:.3f}.png'
            cv2.imwrite(filename, roi)
            print(f"세그먼트 {segment_name} 저장: {filename}")

def main():
    # 숫자 이미지 로드
    img = load_digit_image()
    if img is None:
        return
    
    print(f"이미지 크기: {img.shape}")
    
    # 이미지 전처리
    gray, binary, cleaned = preprocess_digit_image(img)
    
    # 전처리 결과 저장
    cv2.imwrite('debug_gray.png', gray)
    cv2.imwrite('debug_binary.png', binary)
    cv2.imwrite('debug_cleaned.png', cleaned)
    
    # 7세그먼트 영역 정의
    segments = define_seven_segment_regions(cleaned)
    
    print("\n=== 세그먼트 활성화 분석 ===")
    # 각 세그먼트 활성화 여부 확인
    segments_activation = {}
    for segment_name, region in segments.items():
        is_active, ratio = check_segment_activation(cleaned, region, threshold=0.1)
        segments_activation[segment_name] = (is_active, ratio)
        print(f"세그먼트 {segment_name}: 활성화={is_active}, 비율={ratio:.3f}")
    
    # 각 세그먼트를 개별 이미지로 저장
    print("\n=== 개별 세그먼트 저장 ===")
    save_individual_segments(cleaned, segments, segments_activation)
    
    # 숫자 인식 (상세 분석)
    print("\n=== 숫자 인식 상세 분석 ===")
    recognized_digit, best_score, all_results = recognize_digit_detailed(segments_activation)
    
    print(f"\n최종 인식 결과: {recognized_digit} (점수: {best_score})")
    
    # 결과 시각화
    result_img = visualize_segments_detailed(cleaned, segments, segments_activation)
    
    # 결과 이미지에 인식된 숫자 표시
    cv2.putText(result_img, f"Recognized: {recognized_digit}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imwrite('debug_recognition_result.png', result_img)
    print("\n결과 이미지 저장: debug_recognition_result.png")
    
    # 3과 8의 패턴 비교
    print("\n=== 3과 8 패턴 비교 ===")
    print("숫자 3 패턴: ['a', 'b', 'c', 'd', 'g']")
    print("숫자 8 패턴: ['a', 'b', 'c', 'd', 'e', 'f', 'g']")
    print("차이점: 8은 'e', 'f' 세그먼트가 추가로 필요")
    
    # 임계값 조정 테스트
    print("\n=== 임계값 조정 테스트 ===")
    for threshold in [0.05, 0.1, 0.15, 0.2]:
        print(f"\n임계값 {threshold}:")
        test_activation = {}
        for segment_name, region in segments.items():
            is_active, ratio = check_segment_activation(cleaned, region, threshold=threshold)
            test_activation[segment_name] = (is_active, ratio)
            if is_active:
                print(f"  {segment_name}: {ratio:.3f}")

if __name__ == "__main__":
    main() 