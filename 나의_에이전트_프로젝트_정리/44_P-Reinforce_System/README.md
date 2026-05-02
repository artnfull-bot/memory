# 🧠 P-Reinforce: The Autonomous Knowledge Gardener 🚀

> [!IMPORTANT]
> **2026-04-12 업데이트**: 메인 에이전트 프로젝트 정리(Obsidian Vault)에 성공적으로 통합되었습니다.
> - **통합 위치**: `C:\Users\User\Documents\Obsidian Vault\나의_에이전트_프로젝트_정리\`
> - **연결 상태**: 메인 전략 허브 및 타 에이전트들과 시각적 그래프 통합 완료.

**P-Reinforce**는 Andre Karpathy의 **LLM-Wiki 아키텍처**와 **강화학습(Reinforcement Learning)** 이론을 결합하여 파편화된 정보를 영속적이고 유기적인 지식 위키로 자동 변환하는 지식 자동화 에이전트 시스템입니다.

---

## 📌 핵심 미션 (Core Mission)
사용자가 던지는 불완전하고 파편화된 원시 데이터를 실시간으로 모니터링하여, 다음 가치를 창출합니다:
1. **의미론적 분류 (Semantic Classification)**: 데이터의 맥락을 분석하여 최적의 위치를 스스로 결정.
2. **지식 연결 (Graph Linking)**: [[쌍방향 링크]]를 통해 지식 간의 유기적 관계망(Graph) 구축.
3. **지식 강화 (Knowledge Reinforcement)**: 강화학습 보상 정책($R$)을 통해 분류 정확도 및 사용자 만족도를 지속적으로 최적화.
4. **버전 관리 (GitHub Synchronization)**: 모든 지식의 변화를 GitHub에 커밋하여 소중한 지식의 타임라인 기록.

---

## 📁 표준 폴더 구조 (The Structure)
에이전트가 관리하는 폴더의 위계와 역할입니다:

```plaintext
root/
├── 00_Raw/                 # [불변] 사용자가 입력한 가공되지 않은 모든 원본 데이터
│   └── YYYY-MM-DD/         # 날짜별 원본 보관 (Source of Truth)
│
├── 10_Wiki/                # [자동 구조화] 에이전트가 RL 정책에 따라 관리하는 지식 층
│   ├── 🛠️ Projects/        # 목표 중심 (현재 진행 중인 일, 프로젝트별 요약)
│   ├── 💡 Topics/          # 개념 중심 (심리학, 코딩, 철학 등 스스로 생성한 분류)
│   ├── ⚖️ Decisions/       # 의사결정 중심 (왜 이렇게 판단했는가에 대한 기록)
│   └── 🚀 Skills/          # 실행 중심 (프롬프트, 워크플로우 패턴 등)
│
├── 20_Meta/                # [시스템] 지식 엔진의 두뇌 데이터
│   ├── Graph.json          # 지식 간 연결 관계 데이터 (시각화용)
│   ├── Policy.md           # 사용자 피드백이 반영된 정책 (RL Weights)
│   └── Index.md            # 위키의 관문 (Table of Contents)
│
└── .github/                # GitHub Sync 및 자동화 워크플로우
```

---

## 🧠 강화학습 기반 구조화 로직 (The RL Logic)
지식 배치 시 아래 보상 함수 $R$을 극대화하도록 설계되었습니다:

$$R = w_1(\text{Categorization Accuracy}) + w_2(\text{Graph Connectivity}) + w_3(\text{User Satisfaction})$$

- **상태 분석(State)**: `10_Wiki/` 의 구조와 `20_Meta/Graph.json`의 연결망 파악.
- **분류 행동(Action)**: 유사도 85% 이상 시 기존 폴더 배치, 신규모임 시 즉시 새 카테고리 도출.
- **지식 합성**: Karpathy의 '영속적 위키' 템플릿에 맞게 내용을 정제하고 최소 2개 이상의 연관 지식 연결.
- **정책 업데이트(Update)**: 사용자의 칭찬이나 수정 피드백을 수집하여 `20_Meta/Policy.md`에 반영.

---

## 📝 지식 문서 변환 규격 (Wiki Template)
모든 문서는 아래와 같이 정제되어 보관됩니다:

```markdown
---
id: {{UUID}}
category: "[[10_Wiki/Path/To/Folder]]"
confidence_score: 0.0 ~ 1.0 (RL 기반 확신도)
tags: [tag1, tag2]
last_reinforced: YYYY-MM-DD
github_commit: "{{commit_hash}}"
---

# [[문서 제목]]

## 📌 한 줄 통찰 (Summary)
> 이 지식의 핵심을 꿰뚫는 한 문장.

## 📖 구조화된 지식
- 추출된 패턴 및 세부 내용 요약.

## 🔗 지식 연결 (Graph)
- **Parent**: [[상위_카테고리]]
- **Related**: [[연관_기념_A]], [[연관_개념_B]]
- **Raw Source**: [[00_Raw/Original_Note]]
```

---

## 💡 사용자 가이드: "어떻게 에이전트를 가르칠 것인가?"
당신의 피드백이 P-Reinforce를 더 똑똑하게 만듭니다:

1. **칭찬**: "이 폴더 분류 완벽해." → 해당 주제의 유사도 가중치를 높입니다.
2. **수정**: "이건 '코딩'이 아니라 '비즈니스'야." → 두 주제 사이의 경계선을 재설정(Boundary Shift)합니다.
3. **방치**: 에이전트가 만든 구조를 그대로 사용한다면 암묵적 보상으로 간주되어 정책이 정교화됩니다.

---
**Maintained by**: Antigravity (Advanced AI Agent)
**Project Repo**: [jay-s-wiki](https://github.com/artnfull-bot/jay-s-wiki.git)
