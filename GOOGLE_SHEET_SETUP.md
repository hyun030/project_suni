# Google Sheet 설정 가이드

## 구글 시트 공유 설정 방법

현재 구글 시트가 개인 계정에서만 접근 가능한 상태입니다. 애플리케이션이 뉴스 데이터를 가져올 수 있도록 다음 단계를 따라 설정해주세요:

### 방법 1: 서비스 계정으로 공유 (권장)

1. **Google Cloud Console에서 서비스 계정 생성**
   - [Google Cloud Console](https://console.cloud.google.com/)에 접속
   - 새 프로젝트 생성 또는 기존 프로젝트 선택
   - "IAM 및 관리" > "서비스 계정" 메뉴로 이동
   - "서비스 계정 만들기" 클릭
   - 이름 입력 (예: "sk-energy-news-collector")
   - "키 만들기" > "JSON" 선택하여 키 파일 다운로드

2. **구글 시트에 서비스 계정 이메일 공유**
   - 구글 시트에서 "공유" 버튼 클릭
   - 서비스 계정 이메일 주소 입력 (예: `sk-energy-news-collector@project-id.iam.gserviceaccount.com`)
   - "편집자" 권한 부여
   - "완료" 클릭

3. **config.py에 서비스 계정 정보 추가**
   ```python
   GOOGLE_SERVICE_ACCOUNT_JSON = {
       "type": "service_account",
       "project_id": "your-project-id",
       "private_key_id": "...",
       "private_key": "...",
       "client_email": "...",
       "client_id": "...",
       # ... 다운로드한 JSON 파일의 모든 내용
   }
   ```

### 방법 2: 임시로 공개 설정 (테스트용)

1. **구글 시트에서 "공유" 버튼 클릭**
2. **"링크가 있는 모든 사용자" 선택**
3. **"편집자" 권한으로 설정**
4. **"완료" 클릭**

⚠️ **주의**: 방법 2는 보안상 권장되지 않습니다. 테스트 후에는 반드시 방법 1로 변경하거나 접근 권한을 제거하세요.

### 시트 구조 확인

구글 시트는 다음 컬럼을 포함해야 합니다:
- `날짜` (또는 `Date`)
- `회사` (또는 `Company`)
- `제목` (또는 `Title`)
- `키워드` (또는 `Keywords`)
- `출처` (또는 `Source`)
- `URL`

### 문제 해결

- **"권한 없음" 오류**: 서비스 계정 이메일이 시트에 공유되었는지 확인
- **"시트를 찾을 수 없음" 오류**: SHEET_ID가 올바른지 확인 (URL에서 추출)
- **"빈 데이터" 오류**: 시트에 데이터가 있고 첫 번째 행이 헤더인지 확인
