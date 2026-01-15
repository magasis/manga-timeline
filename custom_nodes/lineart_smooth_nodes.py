"""
LineArt Smoothing Nodes - 線画スムージング専用ノード
Yu's Custom Nodes for ComfyUI

モルフォロジー処理、アンチエイリアス、閾値調整を統合した線画処理ノード群
"""

import torch
import numpy as np
import cv2
from PIL import Image


def tensor2np(tensor):
    """Tensor (B,H,W,C) -> numpy array (H,W,C) for first image"""
    if tensor.dim() == 4:
        img = tensor[0].cpu().numpy()
    else:
        img = tensor.cpu().numpy()
    return (img * 255).clip(0, 255).astype(np.uint8)


def np2tensor(img):
    """numpy array (H,W,C) or (H,W) -> Tensor (1,H,W,C)"""
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)


def batch_process(tensor, func):
    """バッチ処理ヘルパー"""
    results = []
    for i in range(tensor.shape[0]):
        img = (tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        processed = func(img)
        if processed.ndim == 2:
            processed = np.stack([processed, processed, processed], axis=-1)
        results.append(torch.from_numpy(processed.astype(np.float32) / 255.0))
    return torch.stack(results, dim=0)


class LineArtSmoother:
    """
    ジャギー線のスムージング専用ノード
    Bilateral/Gaussian/Median フィルタとモルフォロジー処理を組み合わせ
    """
    
    SMOOTH_METHODS = ["bilateral", "gaussian", "median", "none"]
    MORPHOLOGY_OPS = ["none", "erode", "dilate", "open", "close"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "smooth_method": (cls.SMOOTH_METHODS, {"default": "bilateral"}),
                "smooth_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "edge_preserve": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "morphology_op": (cls.MORPHOLOGY_OPS, {"default": "none"}),
                "morphology_iterations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 9, "step": 2}),
                "invert": ("BOOLEAN", {"default": True}),  # デフォルト: 白背景に黒線
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "smooth"
    CATEGORY = "Yu Custom/LineArt"
    
    def smooth(self, image, smooth_method, smooth_strength, edge_preserve, 
               morphology_op, morphology_iterations, kernel_size, invert):
        
        def process_single(img):
            # グレースケール変換（処理用）
            if img.shape[-1] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img[:, :, 0]
            
            result = gray.copy()
            
            # スムージング処理
            if smooth_method == "bilateral" and smooth_strength > 0:
                d = int(smooth_strength * 2) + 1
                sigma_color = int(75 * (1 - edge_preserve) + 10)
                sigma_space = int(75 * smooth_strength / 10 + 10)
                result = cv2.bilateralFilter(result, d, sigma_color, sigma_space)
                
            elif smooth_method == "gaussian" and smooth_strength > 0:
                ksize = int(smooth_strength * 2) | 1  # 奇数に
                ksize = max(1, ksize)
                result = cv2.GaussianBlur(result, (ksize, ksize), 0)
                
            elif smooth_method == "median" and smooth_strength > 0:
                ksize = int(smooth_strength) | 1  # 奇数に
                ksize = max(3, ksize)
                result = cv2.medianBlur(result, ksize)
            
            # モルフォロジー処理
            if morphology_op != "none":
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
                for _ in range(morphology_iterations):
                    if morphology_op == "erode":
                        result = cv2.erode(result, kernel)
                    elif morphology_op == "dilate":
                        result = cv2.dilate(result, kernel)
                    elif morphology_op == "open":
                        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
                    elif morphology_op == "close":
                        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            
            # 反転（デフォルトで白背景に黒線）
            if invert:
                result = 255 - result
            
            return result
        
        output = batch_process(image, process_single)
        return (output,)


class LineArtThreshold:
    """
    閾値調整＆二値化ノード
    Simple/Otsu/Adaptive 閾値と反転・アンチエイリアスオプション
    """
    
    THRESHOLD_METHODS = ["simple", "otsu", "adaptive_mean", "adaptive_gaussian"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "method": (cls.THRESHOLD_METHODS, {"default": "simple"}),
                "invert": ("BOOLEAN", {"default": True}),  # デフォルト: 白背景に黒線
                "antialiasing": ("BOOLEAN", {"default": True}),
                "adaptive_block_size": ("INT", {"default": 11, "min": 3, "max": 51, "step": 2}),
                "adaptive_c": ("INT", {"default": 2, "min": -20, "max": 20, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "threshold_image"
    CATEGORY = "Yu Custom/LineArt"
    
    def threshold_image(self, image, threshold, method, invert, antialiasing,
                        adaptive_block_size, adaptive_c):
        
        def process_single(img):
            # 元のサイズを保存
            orig_h, orig_w = img.shape[:2]
            
            # グレースケール変換
            if img.shape[-1] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img[:, :, 0]
            
            # アンチエイリアス用に一度拡大
            if antialiasing:
                gray = cv2.resize(gray, (orig_w * 2, orig_h * 2), interpolation=cv2.INTER_CUBIC)
            
            # 閾値処理
            thresh_val = int(threshold * 255)
            
            if method == "simple":
                _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                
            elif method == "otsu":
                _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            elif method == "adaptive_mean":
                result = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY, adaptive_block_size, adaptive_c
                )
                
            elif method == "adaptive_gaussian":
                result = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, adaptive_block_size, adaptive_c
                )
            else:
                result = gray
            
            # アンチエイリアス用に元サイズに縮小
            if antialiasing:
                result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
            
            # 反転（デフォルトで白背景に黒線）
            if invert:
                result = 255 - result
            
            return result
        
        output = batch_process(image, process_single)
        return (output,)


class LineArtToolbox:
    """
    線画処理統合ツールボックス
    モード切替式で様々な処理を一つのノードで実行
    """
    
    MODES = [
        "Smooth Only (スムーズのみ)",
        "Threshold Only (閾値のみ)",
        "Smooth → Threshold (順次処理)",
        "Threshold → Smooth (逆順処理)",
        "Full Pipeline (抽出→スムーズ→閾値)",
    ]
    
    SMOOTH_METHODS = ["bilateral", "gaussian", "median"]
    THRESHOLD_METHODS = ["simple", "otsu", "adaptive_gaussian"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": cls.MODES[2]}),
                
                # 共通設定
                "output_grayscale": ("BOOLEAN", {"default": True}),
                
                # スムーズ設定
                "smooth_method": (cls.SMOOTH_METHODS, {"default": "bilateral"}),
                "smooth_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 15.0, "step": 0.5}),
                "edge_preserve": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # モルフォロジー設定
                "line_thin": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
                "line_thick": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
                
                # 閾値設定
                "threshold_method": (cls.THRESHOLD_METHODS, {"default": "simple"}),
                "threshold_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_output": ("BOOLEAN", {"default": True}),  # デフォルト: 白背景に黒線
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Yu Custom/LineArt"
    OUTPUT_NODE = True
    
    def process(self, image, mode, output_grayscale, smooth_method, smooth_strength,
                edge_preserve, line_thin, line_thick, threshold_method, 
                threshold_value, invert_output):
        
        def apply_smooth(gray):
            """スムージング適用"""
            if smooth_method == "bilateral" and smooth_strength > 0:
                d = int(smooth_strength * 2) + 1
                sigma_color = int(75 * (1 - edge_preserve) + 10)
                sigma_space = int(75 * smooth_strength / 10 + 10)
                return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
            elif smooth_method == "gaussian" and smooth_strength > 0:
                ksize = int(smooth_strength * 2) | 1
                return cv2.GaussianBlur(gray, (ksize, ksize), 0)
            elif smooth_method == "median" and smooth_strength > 0:
                ksize = int(smooth_strength) | 1
                ksize = max(3, ksize)
                return cv2.medianBlur(gray, ksize)
            return gray
        
        def apply_threshold(gray):
            """閾値適用"""
            thresh_val = int(threshold_value * 255)
            if threshold_method == "simple":
                _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            elif threshold_method == "otsu":
                _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif threshold_method == "adaptive_gaussian":
                result = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
            return result
        
        def apply_morphology(gray):
            """線の太さ調整"""
            kernel = np.ones((3, 3), np.uint8)
            result = gray.copy()
            
            # 線を細く（erode）
            for _ in range(line_thin):
                result = cv2.erode(result, kernel)
            
            # 線を太く（dilate）
            for _ in range(line_thick):
                result = cv2.dilate(result, kernel)
            
            return result
        
        def process_single(img):
            # 元のサイズを保存（アスペクト比保持確認用）
            orig_h, orig_w = img.shape[:2]
            
            # グレースケール変換
            if img.shape[-1] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img[:, :, 0]
            
            result = gray.copy()
            
            # モードに応じて処理
            if "Smooth Only" in mode:
                result = apply_smooth(result)
                result = apply_morphology(result)
                
            elif "Threshold Only" in mode:
                result = apply_threshold(result)
                result = apply_morphology(result)
                
            elif "Smooth → Threshold" in mode:
                result = apply_smooth(result)
                result = apply_threshold(result)
                result = apply_morphology(result)
                
            elif "Threshold → Smooth" in mode:
                result = apply_threshold(result)
                result = apply_smooth(result)
                result = apply_morphology(result)
                
            elif "Full Pipeline" in mode:
                # コントラスト調整
                result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
                result = apply_smooth(result)
                result = apply_threshold(result)
                result = apply_morphology(result)
            
            # 反転
            if invert_output:
                result = 255 - result
            
            return result
        
        output = batch_process(image, process_single)
        return (output,)


class LineArtResizer:
    """
    元画像のサイズに合わせてリサイズするノード
    LineartStandardPreprocessorなどで変わったサイズを元に戻す
    """
    
    RESIZE_METHODS = ["lanczos", "bilinear", "bicubic", "nearest", "area"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # リサイズ対象の画像
                "reference_image": ("IMAGE",),  # サイズ参照元の画像
                "resize_method": (cls.RESIZE_METHODS, {"default": "lanczos"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "size_info",)
    FUNCTION = "resize_to_reference"
    CATEGORY = "Yu Custom/LineArt"
    OUTPUT_NODE = True
    
    def resize_to_reference(self, image, reference_image, resize_method):
        # 参照画像のサイズを取得
        ref_h, ref_w = reference_image.shape[1], reference_image.shape[2]
        
        # 現在のサイズ
        cur_h, cur_w = image.shape[1], image.shape[2]
        
        # サイズ情報
        size_info = f"元: {cur_w}x{cur_h} → 出力: {ref_w}x{ref_h}"
        
        # サイズが同じなら何もしない
        if cur_h == ref_h and cur_w == ref_w:
            return (image, size_info)
        
        # リサイズメソッドのマッピング
        method_map = {
            "lanczos": cv2.INTER_LANCZOS4,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
        }
        interp = method_map.get(resize_method, cv2.INTER_LANCZOS4)
        
        # バッチ処理
        results = []
        for i in range(image.shape[0]):
            img = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            resized = cv2.resize(img, (ref_w, ref_h), interpolation=interp)
            results.append(torch.from_numpy(resized.astype(np.float32) / 255.0))
        
        output = torch.stack(results, dim=0)
        return (output, size_info)


class CrispLineArt:
    """
    Cannyエッジ検出で極細の境界線を抽出
    ぼかし一切なし。純粋なエッジ検出。白背景に黒線を出力。
    """
    
    LINE_THICKNESS = ["極細", "細め", "普通", "太め", "極太"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "line_thickness": (cls.LINE_THICKNESS, {"default": "極細"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("lineart",)
    FUNCTION = "extract"
    CATEGORY = "Yu Custom/LineArt"
    OUTPUT_NODE = True
    
    def extract(self, image, line_thickness):
        # 線の太さに応じたdilate回数（Cannyは1px線なので太くする方向）
        thickness_map = {
            "極細": 0,   # そのまま（1px）
            "細め": 1,   # 少し太く
            "普通": 2,
            "太め": 3,
            "極太": 5,
        }
        dilate_count = thickness_map.get(line_thickness, 0)
        
        def process_single(img):
            # グレースケール変換
            if img.shape[-1] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img[:, :, 0]
            
            # Cannyエッジ検出（ぼかしなし、純粋なエッジ）
            # 低い閾値で細かいエッジも拾う
            edges = cv2.Canny(gray, 50, 150)
            # edges は白いエッジ線（255）、黒い背景（0）
            
            # 線を太くする（必要な場合）
            if dilate_count > 0:
                kernel = np.ones((3, 3), np.uint8)
                for _ in range(dilate_count):
                    edges = cv2.dilate(edges, kernel)
            
            # 反転して「黒い線、白い背景」に
            result = 255 - edges
            
            return result
        
        output = batch_process(image, process_single)
        return (output,)


class CrispLineArtV2:
    """
    V2: AI線画抽出後の後処理用
    白い線（AI抽出出力）も黒い線も両方対応。
    スケルトン化で細線化し、白背景黒線で出力。
    """
    
    LINE_THICKNESS = ["極細", "細め", "普通", "太め", "極太"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "line_thickness": (cls.LINE_THICKNESS, {"default": "極細"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("lineart",)
    FUNCTION = "extract"
    CATEGORY = "Yu Custom/LineArt"
    OUTPUT_NODE = True
    
    def extract(self, image, line_thickness):
        from skimage.morphology import skeletonize
        
        # 線の太さに応じたdilate回数
        thickness_map = {
            "極細": 0,   # 1px
            "細め": 1,
            "普通": 2,
            "太め": 3,
            "極太": 5,
        }
        dilate_count = thickness_map.get(line_thickness, 0)
        
        def process_single(img):
            # グレースケール変換
            if img.shape[-1] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img[:, :, 0]
            
            # 画像の平均輝度で白線か黒線かを判定
            mean_val = np.mean(gray)
            
            if mean_val < 128:
                # 暗い画像 = 白い線がある（AI線画抽出の出力など）
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            else:
                # 明るい画像 = 黒い線がある
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # binary: 線 = 白(255)、背景 = 黒(0)
            
            # スケルトン化（細線化）
            binary_bool = binary > 0
            skeleton_bool = skeletonize(binary_bool)
            skeleton = (skeleton_bool.astype(np.uint8)) * 255
            
            # 線を太くする（必要な場合）
            if dilate_count > 0:
                kernel = np.ones((3, 3), np.uint8)
                for _ in range(dilate_count):
                    skeleton = cv2.dilate(skeleton, kernel)
            
            # 反転して「黒い線、白い背景」に
            result = 255 - skeleton
            
            return result
        
        output = batch_process(image, process_single)
        return (output,)






# ノード登録
NODE_CLASS_MAPPINGS = {
    "LineArtSmoother": LineArtSmoother,
    "LineArtThreshold": LineArtThreshold,
    "LineArtToolbox": LineArtToolbox,
    "LineArtResizer": LineArtResizer,
    "CrispLineArt": CrispLineArt,
    "CrispLineArtV2": CrispLineArtV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LineArtSmoother": "🎨 LineArt Smoother (スムージング)",
    "LineArtThreshold": "🎨 LineArt Threshold (閾値)",
    "LineArtToolbox": "🎨 LineArt Toolbox (統合ツール)",
    "LineArtResizer": "📐 LineArt Resizer (元サイズに復元)",
    "CrispLineArt": "✨ Crisp LineArt V1 (Canny)",
    "CrispLineArtV2": "✨ Crisp LineArt V2",
}
