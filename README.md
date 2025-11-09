# shortest_temporal_path

running with ~28.9k satellites and 61 time slots. Propagation alone is okay, but the next step (building connectivity) would try to compute pairwise distances for n·(n-1)·M ≈ 28,884²·61 ≈ 51,000,000,000 pairs

🛰️ Step 1: Get Satellite Orbit Data (TLEs)

You start with a file from Space-Track.org
 — the site that publishes official satellite orbit data.

Each satellite’s motion in space is described by two short lines of numbers called a TLE (Two-Line Element set).

A TLE tells us where a satellite is in its orbit and how it moves over time.

You downloaded these as a JSON file (tle_recent_30.json) that contains 28k+ satellites and their latest TLEs.

Think of this as “the orbital fingerprint” of every satellite.

🧠 Step 2: Choose Which Satellites to Use

You don’t want to simulate all 28,000 at once — that’s too heavy.

So the script:

Optionally filters by name (e.g., only STARLINK satellites).

Keeps only the most recent TLE for each satellite.

Limits the total number (you capped it to 1,000).

That’s your “subset” of satellites to analyze — like picking a manageable group out of a global fleet.

⏱️ Step 3: Pick a Time Window and Step Size

You set a start and end time, e.g. 2 hours, and a time step — every 2 minutes.

That defines 61 time points (“slots”) between start and end.

Each slot represents a snapshot of the sky at that moment.

📍 Step 4: Simulate Satellite Positions (SGP4 Propagation)

For every satellite and each time slot, you compute its position in space using the SGP4 model — a physics-based formula used worldwide.

It gives you something like:

Satellite 5 at 2025-11-08 00:00:00Z → [x, y, z] in km
Satellite 5 at 2025-11-08 00:02:00Z → [x, y, z] in km
...


So now you know exactly where every satellite is at every moment.

🌐 Step 5: Build a “Temporal Graph”

This is the key concept from your thesis idea.

Imagine every satellite as a node.

Draw an edge (link) between two satellites if they are close enough (e.g., within 900 km).

Do that for each time slot.

So over time, connections appear and disappear as satellites move around.
That’s why it’s called a temporal graph — a network that changes with time.

🔄 Step 6: Create a Binary Connectivity Timeline

For every pair of satellites (say, Sat A → Sat B):

You record a sequence of 1s and 0s for each slot:

[1, 1, 0, 0, 1, 1, 1, 0, ...]


where 1 = “link up” and 0 = “no link”.

If a link stays up for t consecutive 1s (for example, 2 time slots), it’s considered usable for data transfer.

Each of these binary strings becomes part of the temporal graph’s data structure.

🧮 Step 7: Find the Fastest “Temporal Path”

Now comes the “shortest temporal path” part — your thesis algorithm.

We want to know:

“If I start sending a signal from Satellite A, how fast can it reach Satellite B, given that links come and go over time?”

The algorithm:

Starts with all direct one-hop connections.

Expands paths through intermediate satellites, but only when the timing works (you can’t jump to a satellite before its link appears).

Finds the combination that minimizes elapsed time — the fewest total time slots between start and finish.

So it’s like finding the quickest relay path across moving satellites — a space-time version of Dijkstra’s algorithm.

🗺️ Step 8: Print the Shortest Route

Finally, it prints:

The IDs of the satellites used in the path.

When the connection starts and ends.

How many time slots (or minutes) it takes.

Example:

=== Shortest Temporal Path ===
From 5 to 11
Node sequence: [5, 123, 456, 11]
Elapsed time: 8 slots (~16 minutes)


That’s your optimal communication route through the dynamic satellite network.

🧩 In Simple Terms

Here’s the simplest analogy:

Step	Analogy
Load TLEs	Load the latest GPS of each satellite
Filter	Pick only the ones you care about (e.g. Starlink)
Propagate	Watch them move for 2 hours, snapshot every 2 min
Build graph	Note which satellites can “see” each other (within 900 km)
Temporal graph	Links appear/disappear over time
Shortest path	Find the quickest chain of satellites that connects A → B as the network changes
Output	Print who passes the “message” and how long it takes

So — it’s like Google Maps in space, but the roads (links) open and close every couple of minutes as satellites orbit.
