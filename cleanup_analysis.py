#!/usr/bin/env python3
"""
Directory Cleanup Analysis
Analyzes which files are essential vs. temporary/development files
"""

import os
import re

def analyze_file_usage():
    """Analyze which files are essential for the system"""
    
    # Essential files - core system components
    essential_files = {
        # Core system files
        'requirements.txt': 'Dependencies',
        'README.md': 'Documentation',
        'main_example.py': 'Main entry point',
        
        # Module directories (entire directories)
        'parser_module/': 'CSV to DataFrame processing',
        'knowledge_graph_module/': 'Neo4j knowledge graph',
        'chatbot_module/': 'Chatbot and UI',
        
        # Essential data files
        'enhanced_nlu_ground_truth.csv': 'Current NLU ground truth',
        'improved_ir_ground_truth.csv': 'Current IR ground truth',
        
        # Git and container files
        '.git/': 'Version control',
        '.gitignore': 'Git ignore rules',
        '.devcontainer/': 'Development container',
    }
    
    # Development/testing files to move to backup
    development_files = {
        # Test scripts
        'benchmark_test.py': 'Testing script',
        'check_actual_tables.py': 'Development testing',
        'debug_table_extraction.py': 'Debug script',
        'quick_test.py': 'Quick testing',
        'test_*.py': 'All test files',
        'simple_ui_validation.py': 'UI validation script',
        'ui_enhancement_validation.py': 'Enhancement validation',
        'validate_*.py': 'Validation scripts',
        
        # NLU improvement scripts (to be integrated)
        'enhanced_nlu_ground_truth_generator.py': 'NLU ground truth generator',
        'phase2_enhanced_response.py': 'Phase 2 implementation',
        'phase3_enhanced_entities.py': 'Phase 3 implementation', 
        'phase4_enhanced_similarity.py': 'Phase 4 implementation',
        'nlu_critical_analysis.py': 'NLU analysis script',
        
        # Analysis and documentation files
        'detailed_ir_analysis.csv': 'IR analysis results',
        'detailed_nlu_analysis.csv': 'NLU analysis results',
        'ACADEMIC_BENCHMARKING_GUIDE.md': 'Development guide',
        'ENHANCED_IMPLEMENTATION_SUMMARY.md': 'Implementation summary',
        'NLU_IMPROVEMENT_ANALYSIS.md': 'NLU analysis documentation',
        
        # Sample/backup data
        'enhanced_nlu_ground_truth_full.csv': 'Extended ground truth backup',
        'sample_*.csv': 'Sample data files',
        'improved_nlu_ground_truth.csv': 'Old NLU ground truth',
        'sample.py': 'Sample script',
        
        # Existing backup
        'backup/': 'Existing backup directory'
    }
    
    return essential_files, development_files

def scan_current_directory():
    """Scan current directory and categorize files"""
    
    all_files = []
    for root, dirs, files in os.walk('/workspaces/ran_intelligent_management'):
        # Skip .git and __pycache__ directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__']]
        
        for file in files:
            if not file.startswith('.') and not file.endswith('.pyc'):
                rel_path = os.path.relpath(os.path.join(root, file), '/workspaces/ran_intelligent_management')
                all_files.append(rel_path)
    
    return all_files

def create_cleanup_plan():
    """Create a cleanup plan"""
    
    essential_files, development_files = analyze_file_usage()
    current_files = scan_current_directory()
    
    print("📁 Directory Cleanup Analysis")
    print("=" * 50)
    
    # Files to keep
    keep_files = []
    # Files to move to backup
    move_files = []
    # Files to integrate
    integrate_files = []
    
    for file in current_files:
        file_basename = os.path.basename(file)
        file_dir = os.path.dirname(file) + '/' if os.path.dirname(file) else ''
        
        # Check if essential
        is_essential = False
        for essential_pattern in essential_files.keys():
            if (essential_pattern.endswith('/') and file.startswith(essential_pattern)) or \
               (file == essential_pattern) or \
               (file_basename == essential_pattern):
                is_essential = True
                keep_files.append(file)
                break
        
        if not is_essential:
            # Check if it's a development file
            is_development = False
            for dev_pattern in development_files.keys():
                if (dev_pattern.startswith('test_') and file_basename.startswith('test_')) or \
                   (dev_pattern.startswith('validate_') and file_basename.startswith('validate_')) or \
                   (dev_pattern.startswith('sample_') and file_basename.startswith('sample_')) or \
                   (file == dev_pattern) or \
                   (file_basename == dev_pattern):
                    is_development = True
                    
                    # Special handling for phase files - these need integration
                    if file_basename.startswith('phase') and file_basename.endswith('.py'):
                        integrate_files.append(file)
                    else:
                        move_files.append(file)
                    break
            
            if not is_development:
                # Unknown file - analyze manually
                print(f"❓ Unknown file: {file}")
    
    return keep_files, move_files, integrate_files

if __name__ == "__main__":
    keep_files, move_files, integrate_files = create_cleanup_plan()
    
    print("\n✅ Files to KEEP (Essential):")
    for file in sorted(keep_files):
        print(f"   {file}")
    
    print(f"\n📦 Files to MOVE to backup ({len(move_files)} files):")
    for file in sorted(move_files):
        print(f"   {file}")
    
    print(f"\n🔧 Files to INTEGRATE into chatbot_module ({len(integrate_files)} files):")
    for file in sorted(integrate_files):
        print(f"   {file}")
    
    print(f"\n📊 Summary:")
    print(f"   Essential files: {len(keep_files)}")
    print(f"   Files to backup: {len(move_files)}")
    print(f"   Files to integrate: {len(integrate_files)}")
