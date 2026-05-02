---
tags: [프로젝트, 모바일, 앱개발, 자동화]
---
# 📱 연구자동화 모바일 앱 ZeroDopamine
- **프로젝트 개요**: 연구 자동화 시스템을 모바일에서 제어하고, 수익화 모델을 실험하는 React Native 앱.
- **핵심 기술**: React Native, Expo, Firebase, Google AdMob, PayPal WebView.

## 🛠️ 주요 구현 사항
1. **광고 수익화 (AdMob)**:
    - `react-native-google-mobile-ads` 연동.
    - 하단 배너 광고 및 보상형(리워드) 전면 광고 시스템 구축.
2. **결제 시스템 (PayPal)**:
    - WebView 기반의 페이팔 결제/후원 모달 구현.
    - `onNavigationStateChange`를 통한 결제 완료 감지 로직.
3. **배포 프로세스 (Expo EAS)**:
    - `eas-cli`를 활용한 안드로이드 AAB/APK 클라우드 빌드.
    - 구글 플레이 스토어 내부 테스트(Internal Test) 배포 완료.
4. **트러블슈팅**:
    - **권한 문제**: `blockedPermissions`를 사용하여 마이크 권한 강제 제거 및 개인정보 정책 이슈 해결.
    - **심사 대응**: 20명 테스터 14일 비공개 테스트 요건 대응 및 스토어 시각물 등록.

## 📈 현재 상태
- **빌드 버전**: `1.1.0` (EAS Build profile: production)
- **배포 상태**: 구글 플레이 콘솔 알파 트랙 심사 중 (Internal Testing 운영 중)

## 🔗 연결 노트
- **핵심 에이전트**: [[02_연구자동화_에이전트]] (연구 로직) | [[01_레오_에이전트]] (개발 총괄)
- **상세 가이드**: [[ZeroDopamine_Implementation_Guide]] | [[Google_Play_비공개테스트_심사제출_저널]]
- **기획 문서**: [[앱기획_ZeroDopamine]] | [[37_앱만들기]]
