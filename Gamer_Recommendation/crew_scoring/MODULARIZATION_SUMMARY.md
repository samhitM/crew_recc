# Modularized Crew Scoring System

## Overview
Successfully modularized the crew scoring system from two monolithic files (~800-1000 lines each) into a clean, organized, modular architecture with 3-4 modules per scoring system based on functional similarity.

## Impression Scoring Modularization

### Directory Structure:
```
impressionScoring/
├── database/
│   ├── __init__.py
│   └── db_manager.py              # Database operations
├── graph/
│   ├── __init__.py
│   └── graph_manager.py           # Graph construction & metrics
├── scoring/
│   ├── __init__.py
│   └── scoring_manager.py         # Scoring calculations
├── utils/
│   ├── __init__.py
│   └── helpers.py                 # Utility functions
└── main_impression_calculator.py  # Main entry point
```

### Module Responsibilities:

1. **DatabaseManager** (`database/db_manager.py`):
   - Database connections
   - Fetch friendship data
   - Fetch user games data
   - Fetch message counts
   - Fetch user interactions

2. **GraphManager** (`graph/graph_manager.py`):
   - Build friendship graph
   - Calculate graph metrics (PageRank, K-Shell)
   - Graph visualization
   - Edge weight calculations

3. **ScoringManager** (`scoring/scoring_manager.py`):
   - Calculate profile likes
   - Calculate topological scores
   - Prepare feature data
   - Learn feature weights
   - Calculate user feature scores
   - Calculate website impressions
   - Feature normalization

4. **Utils** (`utils/helpers.py`):
   - Score normalization utilities
   - File operations with fallback
   - Helper functions

## Level Scoring Modularization

### Directory Structure:
```
levelScoring/
├── database/
│   ├── __init__.py
│   └── level_db_manager.py        # Database operations
├── graph/
│   ├── __init__.py
│   └── level_graph_manager.py     # Community graph operations
├── scoring/
│   ├── __init__.py
│   └── level_scoring_manager.py   # Scoring calculations
├── clustering/
│   ├── __init__.py
│   └── level_clustering_manager.py # KNN clustering for levels
├── utils/
│   ├── __init__.py
│   └── level_helpers.py           # Utility functions
└── main_level_calculator.py       # Main entry point
```

### Module Responsibilities:

1. **LevelDatabaseManager** (`database/level_db_manager.py`):
   - Database connections
   - Fetch user games data
   - Fetch friendship data for community detection
   - Fetch user interactions
   - Load impression scores from CSV

2. **LevelGraphManager** (`graph/level_graph_manager.py`):
   - Build community graph
   - Calculate link prediction scores
   - Community graph visualization
   - Edge weight calculations for community detection

3. **LevelScoringManager** (`scoring/level_scoring_manager.py`):
   - Calculate gaming activity scores
   - Calculate community scores
   - Calculate bonus factors
   - Calculate composite scores with normalization
   - Feature normalization

4. **LevelClusteringManager** (`clustering/level_clustering_manager.py`):
   - KNN clustering with elbow method
   - Optimal cluster determination
   - Hybrid clustering approach
   - Level assignment logic

5. **Utils** (`utils/level_helpers.py`):
   - File operations with fallback
   - Level distribution validation
   - Helper functions

## Key Features Maintained:

✅ **Individual normalization** of all components (mean=0, std=1)
✅ **No out_degree calculation** (removed as requested)
✅ **Topological score** = PageRank + K-Shell (50% each)
✅ **All normalized values** included in output CSVs
✅ **Level scoring uses normalized impression scores**
✅ **Hybrid KNN clustering** for balanced level assignment
✅ **Graph visualization** saved as PNG files
✅ **Database integration** with proper error handling
✅ **Modular architecture** with clear separation of concerns

## Benefits of Modularization:

1. **Maintainability**: Each module has a single responsibility
2. **Reusability**: Modules can be imported and used independently
3. **Testability**: Each module can be unit tested separately
4. **Readability**: Code is organized by functionality
5. **Scalability**: Easy to add new features or modify existing ones
6. **Debugging**: Issues can be isolated to specific modules

## Usage:

### Impression Scoring:
```bash
cd crew_scoring/impressionScoring
python main_impression_calculator.py
```

### Level Scoring:
```bash
cd crew_scoring/levelScoring
python main_level_calculator.py
```

## Output Files:
- `crew_impressions_revised.csv` - With all normalized scores
- `crew_levels_revised.csv` - With all normalized scores
- `friendship_graph_with_interactions.png` - Graph visualization
- `community_graph_with_interactions.png` - Community graph visualization

The modularized system maintains all functionality while providing a clean, organized codebase that follows software engineering best practices.
