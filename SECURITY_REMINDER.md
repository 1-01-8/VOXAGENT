# ⚠️ 安全警示 · 请务必阅读

> 本仓库首次 commit（`4e53c5d`）**包含 `.env` 文件**，里面有真实 API Key。
> 仓库虽为 private，但密钥已永久写入 git 历史。本文档列出所有必须注意的场景和应对措施。

---

## 一、当前风险清单

| 风险场景 | 后果 | 触发可能性 |
|---------|------|----------|
| GitHub 账号被盗 / token 泄漏 | 密钥立即外泄 | 低，但发生即灾难 |
| 手滑把仓库改为 public | 密钥全网可搜 | 中，一次误操作即可 |
| 邀请协作者（collaborator） | 对方能看到所有历史 commit 里的密钥 | 低 |
| Fork 后被对方私自转 public | 理论上 fork 继承历史 | 极低 |
| 本地 `.env` 密钥被撞库 / log 抓走 | 连带历史里的也等效失效 | 低 |

**关键认知**：git 历史**不会**因为你后来删除 `.env` 而消失。任何人只要 `git log --all -p` 就能翻出历史 commit 里的密钥。

---

## 二、必须做的 3 件事（立即）

### 1. 在 GitHub 仓库设置里锁死可见性

进入 https://github.com/1-01-8/VoxCareAgent/settings → 拉到最底 **Danger Zone** → 确认 **Change visibility** 显示 "Private"（不要点任何东西，只是确认状态）。

### 2. 在 GitHub 账号设置里启用 2FA

https://github.com/settings/security → **Two-factor authentication** → 启用。这是防止账号被盗的最后一道防线。

### 3. 给自己设一个心理红线

> **永远不要**在 GitHub 仓库页面点击 "Change visibility to Public" 按钮，除非我已经执行下方的"密钥轮换 + 历史清理"流程。

---

## 三、什么时候必须做"密钥轮换 + 历史清理"

**任一条件满足就必须执行**：

- [ ] 想把仓库改为 public
- [ ] 想邀请任何协作者
- [ ] 怀疑 GitHub 账号已被盗
- [ ] 怀疑本地机器被入侵
- [ ] 在任何公共场合（博客/会议/视频）展示过 commit history

---

## 四、密钥轮换 + 历史清理 SOP

### Step 1: 先轮换所有密钥（**绝不能跳过**）

因为 git 历史清理只是"擦掉痕迹"，但如果密钥已经在互联网上暴露过一秒，就必须假定它已泄漏。

打开 `.env`，对每一个 key 去对应平台作废并重发：

| Key 前缀 | 平台 | 作废入口 |
|---------|-----|---------|
| `sk-...`（OpenAI） | platform.openai.com | API Keys → Revoke |
| 硅基流动 | siliconflow.cn | 控制台 → API Keys → 删除 |
| `AIza...`（Gemini） | aistudio.google.com | API Keys → Delete |
| `sk-ant-...`（Anthropic） | console.anthropic.com | API Keys → Disable |
| Qdrant Cloud | cloud.qdrant.io | API Keys → Revoke |

### Step 2: 从 git 历史中删除 `.env`

```bash
# 1. 安装 git-filter-repo（一次性）
brew install git-filter-repo

# 2. 进入仓库
cd /Users/xxm/learning/VoxCareAgent

# 3. 先把 .env 加到 .gitignore（若尚未）
echo ".env" >> .gitignore
git add .gitignore && git commit -m "Ignore .env going forward"

# 4. 从所有历史中彻底删除 .env
git filter-repo --path .env --invert-paths --force

# 5. 强制推送（会改写远端历史！）
git push origin main --force
```

### Step 3: 验证清理干净

```bash
# 下面这条命令应该输出空
git log --all --full-history -- .env
```

---

## 五、日常使用时的良好习惯

### ✅ 应该做

- 新增密钥先改 `.env.example`（只写字段名 + 占位说明），再本地 `.env` 填真值
- 每次 commit 前 `git status` 看一眼有没有误加敏感文件
- 用 [gitleaks](https://github.com/gitleaks/gitleaks) 定期扫自己仓库：
  ```bash
  brew install gitleaks
  gitleaks detect --source . --verbose
  ```

### ❌ 不要做

- 不要在任何 commit message / issue / PR 描述里粘贴真实 key
- 不要用 `git add .` 或 `git add -A` 不看就提交
- 不要在 Slack/微信/截图里发完整 `.env` 给别人

---

## 六、紧急应急（发现密钥已泄漏）

如果你发现密钥真的被滥用（账单异常、接到 OpenAI 风险邮件等）：

1. **第一时间作废**：去对应平台 revoke 当前所有 key
2. **看账单**：检查最近 7 天是否有异常调用，截图保留
3. **申请止损**：OpenAI 有 fraud team，邮件 `support@openai.com` 附证据可申请异常扣费返还
4. **再执行 Step 2**：清理 git 历史（此时 key 已作废，主要是防止未来困扰）

---

## 七、快速检查清单（定期自查，比如每月一次）

```
[ ] GitHub 账号 2FA 开着？
[ ] VoxCareAgent 仓库还是 Private？
[ ] 本地 .env 里的 key 是不是本月刚轮换的？
[ ] `git log --all -- .env` 是否仍然有历史记录？（答案现在是"是"，轮换后应该改成"否"）
[ ] 有没有协作者被意外邀请进来？（https://github.com/1-01-8/VoxCareAgent/settings/access）
```

---

**记住**：现在不处理，是因为账号安全 + 私有仓库双重保护让风险可控。但这不是"没事"，这是"暂时没事"。
