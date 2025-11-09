# 🛰️ Shortest Temporal Path

This project calculates the **fastest time-respecting communication route** between satellites in orbit — a *temporal shortest path* problem.

It uses real-world orbital data (TLEs) from [Space-Track.org](https://www.space-track.org) and simulates how satellite connections appear and disappear over time.

---

## ⚙️ Overview

- Running with **~28.9k satellites** and **61 time slots** (2-minute intervals).  
- Propagation (position calculation) is feasible, but computing pairwise distances for all satellites would mean:

n·(n-1)·M ≈ 28,884²·61 ≈ 51,000,000,000 pairs


That’s why we limit the analysis to a smaller subset (e.g., 1,000 satellites).

---

## 🛰️ Step 1: Get Satellite Orbit Data (TLEs)

- Source: [Space-Track.org](https://www.space-track.org)
- Each satellite’s motion is described by two lines of numbers called a **TLE (Two-Line Element set)**.
- A TLE provides the orbital parameters that define where the satellite is and how it moves.
- You downloaded these as a JSON file (`tle_recent_30.json`) containing 28k+ satellites and their latest TLEs.

> Think of this as the **“orbital fingerprint”** of every satellite.

---

## 🧠 Step 2: Choose Which Satellites to Use

Simulating all 28,000 satellites is too heavy.  
So the script:

- Optionally **filters** by name (e.g., `"STARLINK"`).
- Keeps only the **latest TLE** per satellite.
- **Limits** the total number to a manageable size (default: 1,000).

This gives you a **subset of satellites** — a manageable sample of the global fleet.

---

## ⏱️ Step 3: Pick a Time Window and Step Size

- Define a **start** and **end** time (e.g., 2 hours).
- Choose a **step size** (e.g., every 2 minutes).

That produces **61 time points (slots)** between start and end.  
Each slot represents a snapshot of the sky at that moment.

---

## 📍 Step 4: Simulate Satellite Positions (SGP4 Propagation)

For each satellite and each time slot:

- Compute its 3D position using the **SGP4 model**, a standard physics-based orbit propagator.

Example:
Satellite 5 at 2025-11-08 00:00:00Z → [x, y, z] km
Satellite 5 at 2025-11-08 00:02:00Z → [x, y, z] km


Now you know where every satellite is at each moment in time.

---

## 🌐 Step 5: Build a Temporal Graph

This is the core concept from the **temporal graph** idea in your thesis.

- Each **satellite = a node**.
- Draw an **edge (link)** between two satellites if they are within a chosen distance (e.g., ≤900 km).
- Repeat for each time slot.

Over time, links **appear and disappear** as satellites move — creating a **time-varying graph**.

---

## 🔄 Step 6: Create a Binary Connectivity Timeline

For every satellite pair (e.g., Sat A → Sat B):

- Record a **sequence of 1s and 0s** across time slots:
[1, 1, 0, 0, 1, 1, 1, 0, ...]


- `1` = link up (within range)
- `0` = no link

- A link must remain “up” for at least **t consecutive 1s** (e.g., 2 time slots) to count as stable and usable.

Each binary string becomes a **connectivity timeline** for that edge.

---

## 🧮 Step 7: Find the Fastest “Temporal Path”

Now comes the **temporal shortest path** algorithm — the heart of the thesis.

We ask:

> “If I start sending a signal from Satellite A, how quickly can it reach Satellite B, given that links appear and disappear over time?”

Algorithm steps:

1. Start with all direct one-hop connections.
2. Expand paths through intermediate satellites — but only when timing allows (can’t use a link before it exists).
3. Pick the combination that minimizes **elapsed time** — the fewest total time slots between start and finish.

It’s like finding the quickest relay path across moving satellites —  
a **space-time version of Dijkstra’s algorithm**.

---

## 🗺️ Step 8: Print the Shortest Route

The output includes:

- Source and destination satellite IDs
- The node sequence used
- Start and finish slots/timestamps
- Elapsed duration (in slots and minutes)

Example:
=== Shortest Temporal Path ===
From 5 to 11
Node sequence: [5, 123, 456, 11]
Elapsed time: 8 slots (~16 minutes)


This represents your **optimal communication route** through the dynamic satellite network.

---

## 🧩 In Simple Terms

| Step | Analogy |
|------|----------|
| Load TLEs | Load the latest GPS data of each satellite |
| Filter | Pick only the ones you care about (e.g., Starlink) |
| Propagate | Watch them move for 2 hours, snapshot every 2 min |
| Build Graph | Note which satellites can “see” each other (within 900 km) |
| Temporal Graph | Links appear/disappear over time |
| Shortest Path | Find the quickest chain of satellites from A → B as the network changes |
| Output | Print who passes the “message” and how long it takes |

> It’s like **Google Maps for space** — but the roads (links) open and close every few minutes as satellites orbit.

---

## ⚡ Performance Note

For 1,000 satellites and 61 slots, the vectorized code handles about **61 million pairwise distance checks**, which is reasonable.  
For 28k satellites, it would be **~51 billion checks** — impossible without distributed computing or pruning (filtering).

---

## 📚 References

- **Data source:** [Space-Track.org](https://www.space-track.org)
- **Propagation model:** [SGP4](https://pypi.org/project/sgp4/) (Vallado/AFRL)
- **Concept:** “All-Pairs Shortest Temporal Path” — based on your thesis algorithm.

---

## 🧠 TL;DR

We build a **time-aware satellite network** from real orbital data and use it to find the **fastest possible path** a signal could take from one satellite to another —  
while the network itself keeps changing as the satellites move around Earth.





