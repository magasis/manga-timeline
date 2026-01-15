"""
Yu's Custom Nodes for ComfyUI
- TextConcatDisplay: テキストを結合し、結果を表示しつつSTRINGを出力
- TagExtractAndCombine: 画像からタグを抽出し、ベースプロンプトと結合
- LineArt Nodes: 線画スムージング・閾値調整ノード
- ChatGPTPoseTransfer: ChatGPT API経由で2画像からポーズ転写画像を生成
"""

from .nodes import NODE_CLASS_MAPPINGS as NODES_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY
from .lineart_smooth_nodes import NODE_CLASS_MAPPINGS as LINEART_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LINEART_DISPLAY
from .chatgpt_image_node import NODE_CLASS_MAPPINGS as CHATGPT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CHATGPT_DISPLAY

# すべてのマッピングを統合
NODE_CLASS_MAPPINGS = {**NODES_MAPPINGS, **LINEART_MAPPINGS, **CHATGPT_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODES_DISPLAY, **LINEART_DISPLAY, **CHATGPT_DISPLAY}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
