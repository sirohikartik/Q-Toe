// Grid elements
const r11 = document.getElementById("11");
const r12 = document.getElementById("12");
const r13 = document.getElementById("13");

const r21 = document.getElementById("21");
const r22 = document.getElementById("22");
const r23 = document.getElementById("23");

const r31 = document.getElementById("31");
const r32 = document.getElementById("32");
const r33 = document.getElementById("33");

const boxes = [r11, r12, r13, r21, r22, r23, r31, r32, r33];

// UI elements
const statusBanner = document.getElementById("status-banner");
const modeSelect = document.getElementById("opponent-mode");
const scorePlayerEl = document.getElementById("score-player");
const scoreTiesEl = document.getElementById("score-ties");
const scoreAiEl = document.getElementById("score-ai");
const btnRestart = document.getElementById("btn-restart");
const btnResetScores = document.getElementById("btn-reset-scores");
const policyStatusEl = document.getElementById("policy-status");

// Game state
let record = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
];
let filled = new Array(9).fill(0);
let gameOver = false;
let isAiThinking = false;

let scores = {
    player: 0,
    ties: 0,
    ai: 0
};

// Trained RL Policy Table loaded from training pipeline
let trainedPolicy = null;

async function loadTrainedPolicy() {
    try {
        const response = await fetch("trained_policy.json");
        if (response.ok) {
            trainedPolicy = await response.json();
            const stateCount = Object.keys(trainedPolicy).length;
            policyStatusEl.textContent = `DQN Policy Active (${stateCount} states)`;
            policyStatusEl.style.color = "#3fb950";
        } else {
            policyStatusEl.textContent = "Policy fallback (Minimax engine)";
        }
    } catch (e) {
        policyStatusEl.textContent = "Policy fallback (Minimax engine)";
    }
}
loadTrainedPolicy();

// Attach cell listeners
boxes.forEach((box, index) => {
    box.addEventListener("click", () => handleCellClick(index));
});

btnRestart.addEventListener("click", resetBoard);
btnResetScores.addEventListener("click", () => {
    scores = { player: 0, ties: 0, ai: 0 };
    updateScoreUI();
});

function handleCellClick(index) {
    if (filled[index] === 1 || gameOver || isAiThinking) return;

    // Player move (X = 1)
    makeMove(index, "X", 1);

    const winner = checkWinner();
    if (winner !== null) {
        handleGameOver(winner);
        return;
    }

    // AI Turn (O = 2)
    isAiThinking = true;
    updateStatus("AI is thinking...", "ai-turn");

    setTimeout(() => {
        aiMove();
        const aiWinner = checkWinner();
        if (aiWinner !== null) {
            handleGameOver(aiWinner);
        } else {
            isAiThinking = false;
            updateStatus("Your Turn (X)", "");
        }
    }, 280);
}

function makeMove(index, symbol, playerCode) {
    boxes[index].textContent = symbol;
    boxes[index].classList.add(symbol.toLowerCase());
    filled[index] = 1;

    const row = Math.floor(index / 3);
    const col = index % 3;
    record[row][col] = playerCode;
}

function getAvailableIndices() {
    const indices = [];
    for (let i = 0; i < 9; i++) {
        if (filled[i] === 0) indices.push(i);
    }
    return indices;
}

function aiMove() {
    const mode = modeSelect.value;
    let chosenIndex = -1;

    if (mode === "rl") {
        chosenIndex = getRlAction();
    } else if (mode === "minimax") {
        chosenIndex = getMinimaxAction();
    } else {
        // Random
        chosenIndex = getRandomAction();
    }

    if (chosenIndex !== -1) {
        makeMove(chosenIndex, "O", 2);
    }
}

// Deep Q-Network Policy Lookup
function getRlAction() {
    if (trainedPolicy) {
        // Encode current board into key: '0' = empty, '1' = X, '2' = O
        const flatBoard = record.flat();
        const stateKey = flatBoard.map(v => v.toString()).join("");

        if (trainedPolicy[stateKey] !== undefined) {
            const action = trainedPolicy[stateKey];
            if (filled[action] === 0) {
                return action;
            }
        }
    }
    // Fallback to Minimax if state not in table or table loading
    return getMinimaxAction();
}

function getRandomAction() {
    const available = getAvailableIndices();
    if (available.length === 0) return -1;
    return available[Math.floor(Math.random() * available.length)];
}

// Minimax Algorithm Implementation
function getMinimaxAction() {
    const available = getAvailableIndices();
    if (available.length === 0) return -1;

    let bestScore = -Infinity;
    let bestMove = available[0];

    for (let index of available) {
        const row = Math.floor(index / 3);
        const col = index % 3;
        record[row][col] = 2; // AI move

        let score = minimax(record, 0, false, -Infinity, Infinity);
        record[row][col] = 0; // backtrack

        if (score > bestScore) {
            bestScore = score;
            bestMove = index;
        }
    }
    return bestMove;
}

function minimax(board, depth, isMaximizing, alpha, beta) {
    const winner = checkWinner();
    if (winner === 2) return 10 - depth;
    if (winner === 1) return depth - 10;
    if (winner === 0) return 0;

    const available = [];
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            if (board[r][c] === 0) available.push([r, c]);
        }
    }

    if (isMaximizing) {
        let maxEval = -Infinity;
        for (let [r, c] of available) {
            board[r][c] = 2;
            let evalScore = minimax(board, depth + 1, false, alpha, beta);
            board[r][c] = 0;
            maxEval = Math.max(maxEval, evalScore);
            alpha = Math.max(alpha, evalScore);
            if (beta <= alpha) break;
        }
        return maxEval;
    } else {
        let minEval = Infinity;
        for (let [r, c] of available) {
            board[r][c] = 1;
            let evalScore = minimax(board, depth + 1, true, alpha, beta);
            board[r][c] = 0;
            minEval = Math.min(minEval, evalScore);
            beta = Math.min(beta, evalScore);
            if (beta <= alpha) break;
        }
        return minEval;
    }
}

// Win, Loss & Draw Evaluation
function checkWinner() {
    // Check rows & columns
    for (let i = 0; i < 3; i++) {
        if (record[i][0] !== 0 && record[i][0] === record[i][1] && record[i][1] === record[i][2]) {
            return record[i][0];
        }
        if (record[0][i] !== 0 && record[0][i] === record[1][i] && record[1][i] === record[2][i]) {
            return record[0][i];
        }
    }

    // Check diagonals
    if (record[0][0] !== 0 && record[0][0] === record[1][1] && record[1][1] === record[2][2]) {
        return record[0][0];
    }
    if (record[0][2] !== 0 && record[0][2] === record[1][1] && record[1][1] === record[2][0]) {
        return record[0][2];
    }

    // Check draw
    if (getAvailableIndices().length === 0) {
        return 0; // Draw
    }

    return null; // Ongoing
}

function handleGameOver(winner) {
    gameOver = true;
    isAiThinking = false;

    if (winner === 1) {
        scores.player++;
        updateStatus("🎉 You Won!", "win");
    } else if (winner === 2) {
        scores.ai++;
        updateStatus("🤖 AI Won!", "loss");
    } else {
        scores.ties++;
        updateStatus("🤝 It's a Draw!", "draw");
    }

    updateScoreUI();
}

function updateStatus(text, className) {
    statusBanner.textContent = text;
    statusBanner.className = "status-banner " + className;
}

function updateScoreUI() {
    scorePlayerEl.textContent = scores.player;
    scoreTiesEl.textContent = scores.ties;
    scoreAiEl.textContent = scores.ai;
}

function resetBoard() {
    record = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ];
    filled.fill(0);
    gameOver = false;
    isAiThinking = false;

    boxes.forEach(box => {
        box.textContent = "";
        box.classList.remove("x", "o");
    });

    updateStatus("Your Turn (X)", "");
}
