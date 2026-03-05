M × N Zero-Sum Game Solver (Java Swing)

The M × N Zero-Sum Game Solver is a Java Swing application that enables users to input, analyze, and solve zero-sum matrix games. The tool allows users to create a payoff matrix and automatically generates its negative counterpart while computing important game theory metrics such as Maximin, Minimax, and the Saddle Point.

This application is designed primarily for students learning Game Theory, providing an interactive way to understand strategic decision-making concepts.

Features

Graphical user interface (GUI) for entering payoff matrices

Supports matrices up to 6 × 6 dimensions

Automatically generates:

Matrix A – user-entered payoff matrix

Matrix B = –A – automatically computed negative matrix

Computes key game theory results:

Maximin value (row player's strategy)

Minimax value (column player's strategy)

Saddle Point detection if it exists

Simple and intuitive interface with buttons for matrix generation and calculations

Developed using Java Swing with absolute layout for precise component placement

How It Works

The user specifies the matrix dimensions M × N (where 
1
≤
𝑀
,
𝑁
≤
6
1≤M,N≤6).

The payoff values for Matrix A are entered manually through the interface.

The application automatically computes Matrix B by negating each element of Matrix A 
(
𝐵
=
−
𝐴
)
(B=−A).

The program calculates:

The Maximin value from Matrix A

The Minimax value from Matrix B

By comparing these values, the program determines whether a Saddle Point exists in the game.

Instructions for Use

Enter the number of rows (M) and columns (N).

Click Generate Matrix Fields to create the input grid.

Fill in the values of Matrix A.

Click Calculate to process the matrix.

The application will display:

Matrix A and Matrix B

Maximin and Minimax values

Saddle Point result (if present)
