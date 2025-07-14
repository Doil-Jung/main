import cv2
import numpy as np

def load_perspective_image():
    """투시 변환된 이미지 로드"""
    img = cv2.imread('perspective_corrected.png')
    if img is None:
        print("perspective_corrected.png 파일을 찾을 수 없습니다.")
        return None
    return img

def preprocess_image(img):
    """이미지 전처리 - 7세그먼트 디스플레이용"""
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 적응형 이진화 (조명 변화에 강함)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return gray, binary, cleaned

def find_digit_regions_method1_contours(img):
    """방법 1: 윤곽선 검출로 숫자 영역 찾기"""
    # 윤곽선 찾기
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 면적이 일정 크기 이상인 윤곽선만 필터링
    min_area = 100  # 최소 면적
    max_area = img.shape[0] * img.shape[1] // 4  # 최대 면적 (전체의 1/4)
    
    digit_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            # 7세그먼트 숫자는 보통 세로로 긴 형태
            if 0.3 < aspect_ratio < 2.0:
                digit_regions.append((x, y, w, h))
    
    return digit_regions

def find_digit_regions_method2_projection(img):
    """방법 2: 수평/수직 투영으로 숫자 영역 찾기"""
    h, w = img.shape
    
    # 수평 투영 (가로 방향 픽셀 분포)
    horizontal_projection = np.sum(img, axis=0)
    
    # 수직 투영 (세로 방향 픽셀 분포)
    vertical_projection = np.sum(img, axis=1)
    
    # 임계값으로 숫자 영역 찾기
    h_threshold = np.max(horizontal_projection) * 0.1
    v_threshold = np.max(vertical_projection) * 0.1
    
    # 수평 방향에서 숫자 경계 찾기
    h_regions = []
    in_digit = False
    start_x = 0
    
    for x in range(w):
        if horizontal_projection[x] > h_threshold and not in_digit:
            start_x = x
            in_digit = True
        elif horizontal_projection[x] <= h_threshold and in_digit:
            end_x = x
            if end_x - start_x > 10:  # 최소 너비
                h_regions.append((start_x, end_x))
            in_digit = False
    
    # 수직 방향에서 숫자 경계 찾기
    v_regions = []
    in_digit = False
    start_y = 0
    
    for y in range(h):
        if vertical_projection[y] > v_threshold and not in_digit:
            start_y = y
            in_digit = True
        elif vertical_projection[y] <= v_threshold and in_digit:
            end_y = y
            if end_y - start_y > 10:  # 최소 높이
                v_regions.append((start_y, end_y))
            in_digit = False
    
    # 수평/수직 영역 조합
    if h_regions and v_regions:
        y_start, y_end = v_regions[0]  # 첫 번째 수직 영역 사용
        digit_regions = []
        for x_start, x_end in h_regions:
            digit_regions.append((x_start, y_start, x_end - x_start, y_end - y_start))
        return digit_regions
    
    return []

def find_digit_regions_method3_connected_components(img):
    """방법 3: 연결 요소 분석으로 숫자 영역 찾기"""
    # 연결 요소 분석
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    
    digit_regions = []
    for i in range(1, num_labels):  # 0번은 배경
        x, y, w, h, area = stats[i]
        
        # 면적 필터링
        if area < 100 or area > img.shape[0] * img.shape[1] // 4:
            continue
        
        # 종횡비 필터링
        aspect_ratio = w / h
        if 0.3 < aspect_ratio < 2.0:
            digit_regions.append((x, y, w, h))
    
    return digit_regions

def visualize_results(img, regions, method_name):
    """결과 시각화"""
    result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for i, (x, y, w, h) in enumerate(regions):
        # 사각형 그리기
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 번호 표시
        cv2.putText(result_img, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 제목 추가
    cv2.putText(result_img, method_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return result_img

def main():
    # 투시 변환된 이미지 로드
    img = load_perspective_image()
    if img is None:
        return
    
    print(f"이미지 크기: {img.shape}")
    
    # 이미지 전처리
    gray, binary, cleaned = preprocess_image(img)
    
    # 전처리 결과 저장
    cv2.imwrite('preprocessed_gray.png', gray)
    cv2.imwrite('preprocessed_binary.png', binary)
    cv2.imwrite('preprocessed_cleaned.png', cleaned)
    
    # 각 방법으로 숫자 영역 찾기
    methods = [
        ("Contour Detection", find_digit_regions_method1_contours),
        ("Projection Analysis", find_digit_regions_method2_projection),
        ("Connected Components", find_digit_regions_method3_connected_components)
    ]
    
    results = []
    for method_name, method_func in methods:
        regions = method_func(cleaned)
        results.append((method_name, regions))
        print(f"{method_name}: {len(regions)}개 영역 발견")
        for i, (x, y, w, h) in enumerate(regions):
            print(f"  영역 {i+1}: ({x}, {y}) 크기 {w}x{h}")
    
    # 결과 시각화 및 저장
    for method_name, regions in results:
        result_img = visualize_results(cleaned, regions, method_name)
        filename = f'digit_detection_{method_name.lower().replace(" ", "_")}.png'
        cv2.imwrite(filename, result_img)
        print(f"{filename} 저장됨")
        
        # 각 숫자 영역을 개별 이미지로 저장
        for i, (x, y, w, h) in enumerate(regions):
            digit_img = cleaned[y:y+h, x:x+w]
            digit_filename = f'digit_{method_name.lower().replace(" ", "_")}_{i+1}.png'
            cv2.imwrite(digit_filename, digit_img)
            print(f"  {digit_filename} 저장됨 (크기: {w}x{h})")

if __name__ == "__main__":
    main() 