import matplotlib.pyplot as plt
import pandas as pd

def create_comparison_charts():
    """Create visualizations from the hybrid evaluation results"""
    
    # Data from our evaluation
    methods = ['Semantic', 'Hybrid', 'Keyword']
    scores = [4.50, 4.00, 3.67]
    times = [8.92, 9.81, 7.16]
    
    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Performance scores
    bars1 = ax1.bar(methods, scores, color=['#4CAF50', '#FFC107', '#F44336'])
    ax1.set_title('Search Method Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Score (/5.0)', fontweight='bold')
    ax1.set_ylim(0, 5.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Response times
    bars2 = ax2.bar(methods, times, color=['#4CAF50', '#FFC107', '#F44336'])
    ax2.set_title('Average Response Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time (seconds)', fontweight='bold')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('hybrid_comparison_charts.png', dpi=300, bbox_inches='tight')
    print(" Comparison charts saved as hybrid_comparison_charts.png")
    
    # Create use case recommendation chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    use_cases = {
        'Semantic': ['Conceptual Questions', 'Complex Analysis', 'Business Risks'],
        'Keyword': ['Exact Terms', 'Specific Numbers', 'Phrase Matching'],
        'Hybrid': ['Mixed Queries', 'When Unsure', 'Balanced Approach']
    }
    
    colors = ['#4CAF50', '#FFC107', '#F44336']
    
    for i, (method, cases) in enumerate(use_cases.items()):
        y_pos = [i * 0.8 + j * 0.25 for j in range(len(cases))]
        ax.barh(y_pos, [1] * len(cases), height=0.2, color=colors[i], label=method)
        
        for j, case in enumerate(cases):
            ax.text(0.5, y_pos[j], case, ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_yticks([i * 0.8 + 0.25 for i in range(3)])
    ax.set_yticklabels(['Semantic', 'Hybrid', 'Keyword'], fontweight='bold')
    ax.set_xlabel('Recommended Use Cases', fontweight='bold')
    ax.set_title('Optimal Search Method by Query Type', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('search_method_recommendations.png', dpi=300, bbox_inches='tight')
    print(" Recommendation chart saved as search_method_recommendations.png")

if __name__ == "__main__":
    create_comparison_charts()