import Foundation
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers
import AVFoundation

// ============================================================================
// CLAUSEGUARD AI — GOOGLE CLOUD NEXT × 3BLUE1BROWN MINIMALIST EDITION
// 15-Slide High-Resolution Deck (Updated with Executed m1/m2/m3/m4 & Bias Audit):
//   - PNGs rendered at 4K UHD Retina (3840x2160, 2.0x vector scale)
//   - Animated GIFs rendered at Full HD (1920x1080, 72/120 frames @ 8fps)
//     with custom 256-color Dark Stage LUT quantization (zero web-safe dithering!)
//   - MP4 videos rendered at Full HD (1920x1080 H.264)
// Zero slide numbering · Zero em-dashes · Zero trailing periods in titles
// Authentic Google I/O / Cloud Next Divided Cards + Frameless 3B1B Math
// ============================================================================

let W: CGFloat = 1920
let H: CGFloat = 1080

// Register official Google Sans & Google Sans Code fonts
let fontPaths = [
    "/Users/jonathan/Desktop/french-data-rag/docs/slides/fonts/GoogleSans-Bold.ttf",
    "/Users/jonathan/Desktop/french-data-rag/docs/slides/fonts/GoogleSans-Medium.ttf",
    "/Users/jonathan/Desktop/french-data-rag/docs/slides/fonts/GoogleSans-Regular.ttf",
    "/Users/jonathan/Desktop/french-data-rag/docs/slides/fonts/JetBrainsMono-Regular.ttf",
    "/Applications/Gemini.app/Contents/Resources/Resources/GoogleSans_v6.004/Desktop/full/ttf/GoogleSans-Regular.ttf",
    "/Applications/Gemini.app/Contents/Resources/Resources/GoogleSans_v6.004/Desktop/full/ttf/GoogleSans-Medium.ttf",
    "/Applications/Gemini.app/Contents/Resources/Resources/GoogleSans_v6.004/Desktop/full/ttf/GoogleSans-Bold.ttf",
    "/Applications/Gemini.app/Contents/Resources/Resources/GoogleSans_v6.004/Desktop/full/ttf/GoogleSans-Italic.ttf",
    "/Applications/Gemini.app/Contents/Resources/GoogleSansCode-Regular.ttf"
]
for fp in fontPaths {
    let url = URL(fileURLWithPath: fp)
    if FileManager.default.fileExists(atPath: fp) {
        CTFontManagerRegisterFontsForURL(url as CFURL, .process, nil)
    }
}

func rgba(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1.0) -> CGColor {
    return CGColor(srgbRed: r / 255.0, green: g / 255.0, blue: b / 255.0, alpha: a)
}

func withAlpha(_ c: CGColor, _ a: CGFloat) -> CGColor {
    return c.copy(alpha: a) ?? c
}

// ----------------------------------------------------------------------------
// Palette: True #000000 Pitch-Black Stage + Google Cloud Next / 3B1B Accents
// ----------------------------------------------------------------------------
let cBg         = rgba(0, 0, 0)              // #000000 Pure Pitch-Black
let cCardBg     = rgba(3, 5, 9, 0.94)        // #030509 Pure-black Keynote card interior
let cSurface    = rgba(10, 14, 22, 0.88)     // #0A0E16 Subtle dark glass surface
let cSurfaceHi  = rgba(18, 24, 36, 0.92)     // #121824 Elevated surface
let cInk        = rgba(248, 249, 250)        // #F8F9FA Crisp White
let cSlate      = rgba(220, 228, 240)        // #DCE4F0 High-contrast projector silver-white
let cMuted      = rgba(148, 160, 178)        // #94A0B2 Legible muted grey for * footnotes / axes
let cStroke     = rgba(42, 52, 70)           // #2A3446 Subtle border

let cBlue       = rgba(88, 151, 251)         // #5897FB Google Keynote Luminous Blue
let cBlueTint   = rgba(88, 151, 251, 0.13)
let cRed        = rgba(234, 67, 53)          // #EA4335 Google Red
let cCoral      = rgba(255, 107, 107)        // #FF6B6B High-contrast Coral
let cCoralTint  = rgba(255, 107, 107, 0.13)
let cAmber      = rgba(251, 188, 4)          // #FBBC04 Google Yellow / 3B1B Gold
let cAmberTint  = rgba(251, 188, 4, 0.13)
let cGreen      = rgba(68, 186, 101)         // #44BA65 Google Luminous Green
let cGreenTint  = rgba(68, 186, 101, 0.13)
let cCyan       = rgba(78, 205, 196)         // #4ECDC4 3Blue1Brown Teal / Cyan
let cCyanTint   = rgba(78, 205, 196, 0.13)
let cViolet     = rgba(175, 92, 247)         // #AF5CF7 DeepMind Luminous Violet
let cVioletTint = rgba(175, 92, 247, 0.14)

// ----------------------------------------------------------------------------
// Typography Helpers
// ----------------------------------------------------------------------------
enum FontWeight {
    case regular, medium, bold, mono, mathItalic, mathRoman, mathBold
}

func makeFont(_ weight: FontWeight, size: CGFloat) -> CTFont {
    let candidates: [String]
    switch weight {
    case .bold:
        candidates = ["GoogleSans-Bold", "Google Sans Bold", "HelveticaNeue-Bold"]
    case .medium:
        candidates = ["GoogleSans-Medium", "Google Sans Medium", "HelveticaNeue-Medium"]
    case .regular:
        candidates = ["GoogleSans-Regular", "Google Sans", "HelveticaNeue"]
    case .mono:
        candidates = ["JetBrainsMono-Regular", "GoogleSansCode-Regular", "Google Sans Code", "Menlo-Regular"]
    case .mathItalic:
        candidates = ["TimesNewRomanPS-ItalicMT", "Georgia-Italic", "GoogleSans-Italic"]
    case .mathRoman:
        candidates = ["TimesNewRomanPSMT", "Georgia", "GoogleSans-Regular"]
    case .mathBold:
        candidates = ["TimesNewRomanPS-BoldItalicMT", "TimesNewRomanPS-BoldMT", "GoogleSans-Bold"]
    }
    for name in candidates {
        let f = CTFontCreateWithName(name as CFString, size, nil)
        let actual = CTFontCopyPostScriptName(f) as String
        if actual.lowercased().contains(name.lowercased().replacingOccurrences(of: " ", with: "")) || name == candidates.last {
            return f
        }
    }
    return CTFontCreateWithName("HelveticaNeue" as CFString, size, nil)
}

enum TextAlign {
    case left, center, right
}

func measureText(_ text: String, size: CGFloat, weight: FontWeight = .regular, tracking: CGFloat = 0) -> CGSize {
    let font = makeFont(weight, size: size)
    var attrs: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(rawValue: kCTFontAttributeName as String): font
    ]
    if tracking != 0 {
        attrs[NSAttributedString.Key(rawValue: kCTKernAttributeName as String)] = tracking
    }
    let attrStr = NSAttributedString(string: text, attributes: attrs)
    let line = CTLineCreateWithAttributedString(attrStr)
    var ascent: CGFloat = 0
    var descent: CGFloat = 0
    var leading: CGFloat = 0
    let width = CGFloat(CTLineGetTypographicBounds(line, &ascent, &descent, &leading))
    return CGSize(width: width, height: ascent + descent)
}

@discardableResult
func drawText(
    _ ctx: CGContext,
    _ text: String,
    x: CGFloat,
    y: CGFloat,
    size: CGFloat,
    weight: FontWeight = .regular,
    color: CGColor = cInk,
    align: TextAlign = .left,
    tracking: CGFloat = 0
) -> CGFloat {
    let font = makeFont(weight, size: size)
    var attrs: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(rawValue: kCTFontAttributeName as String): font,
        NSAttributedString.Key(rawValue: kCTForegroundColorAttributeName as String): color
    ]
    if tracking != 0 {
        attrs[NSAttributedString.Key(rawValue: kCTKernAttributeName as String)] = tracking
    }
    let attrStr = NSAttributedString(string: text, attributes: attrs)
    let line = CTLineCreateWithAttributedString(attrStr)
    var ascent: CGFloat = 0
    var descent: CGFloat = 0
    var leading: CGFloat = 0
    let width = CGFloat(CTLineGetTypographicBounds(line, &ascent, &descent, &leading))

    var drawX = x
    if align == .center { drawX = x - width / 2.0 }
    else if align == .right { drawX = x - width }

    ctx.saveGState()
    ctx.textPosition = CGPoint(x: drawX, y: y)
    CTLineDraw(line, ctx)
    ctx.restoreGState()
    return width
}

func drawTwoToneTitle(
    _ ctx: CGContext,
    leftText: String,
    rightText: String,
    rightColor: CGColor = cBlue,
    y: CGFloat = 955,
    size: CGFloat = 54,
    centerX: CGFloat = 960,
    tracking: CGFloat = -1.6
) {
    let w1 = measureText(leftText, size: size, weight: .bold, tracking: tracking).width
    let w2 = measureText(rightText, size: size, weight: .bold, tracking: tracking).width
    let totalW = w1 + w2
    let startX = centerX - totalW / 2.0
    drawText(ctx, leftText, x: startX, y: y, size: size, weight: .bold, color: cInk, align: .left, tracking: tracking)
    drawText(ctx, rightText, x: startX + w1, y: y, size: size, weight: .bold, color: rightColor, align: .left, tracking: tracking)
}

// ----------------------------------------------------------------------------
// Native LaTeX-Style Mathematical Formula Renderer
// ----------------------------------------------------------------------------
indirect enum MathToken {
    case i(String, CGColor? = nil)
    case b(String, CGColor? = nil)
    case r(String, CGColor? = nil)
    case op(String, CGColor? = nil)
    case sub(String, CGColor? = nil)
    case sup(String, CGColor? = nil)
    case subSup(sub: String, sup: String, CGColor? = nil)
    case frac(num: [MathToken], den: [MathToken], CGColor? = nil)
    case sqrt([MathToken], CGColor? = nil)
    case space(CGFloat)
}

func measureMathTokens(_ tokens: [MathToken], baseSize: CGFloat) -> CGFloat {
    var totalW: CGFloat = 0
    for tok in tokens {
        switch tok {
        case .i(let s, _):
            totalW += measureText(s, size: baseSize, weight: .mathItalic).width + 1.5
        case .b(let s, _):
            totalW += measureText(s, size: baseSize, weight: .mathBold).width + 1.5
        case .r(let s, _):
            totalW += measureText(s, size: baseSize, weight: .mathRoman).width + 1.5
        case .op(let s, _):
            totalW += measureText(s, size: baseSize * 0.90, weight: .medium).width + 2.0
        case .sub(let s, _):
            totalW += measureText(s, size: baseSize * 0.73, weight: .mathRoman).width + 1.5
        case .sup(let s, _):
            totalW += measureText(s, size: baseSize * 0.73, weight: .mathRoman).width + 1.5
        case .subSup(let subS, let supS, _):
            let wSub = measureText(subS, size: baseSize * 0.71, weight: .mathRoman).width
            let wSup = measureText(supS, size: baseSize * 0.71, weight: .mathRoman).width
            totalW += max(wSub, wSup) + 2.5
        case .frac(let num, let den, _):
            let wNum = measureMathTokens(num, baseSize: baseSize * 0.85)
            let wDen = measureMathTokens(den, baseSize: baseSize * 0.85)
            totalW += max(wNum, wDen) + 12.0
        case .sqrt(let inner, _):
            let wIn = measureMathTokens(inner, baseSize: baseSize * 0.88)
            totalW += wIn + baseSize * 0.62 + 6.0
        case .space(let sp):
            totalW += sp
        }
    }
    return totalW
}

@discardableResult
func drawMath(
    _ ctx: CGContext,
    _ tokens: [MathToken],
    x: CGFloat,
    y: CGFloat,
    size: CGFloat,
    color: CGColor = cInk,
    align: TextAlign = .left
) -> CGFloat {
    let totalW = measureMathTokens(tokens, baseSize: size)
    var curX = x
    if align == .center { curX = x - totalW / 2.0 }
    else if align == .right { curX = x - totalW }

    for tok in tokens {
        switch tok {
        case .i(let s, let c):
            let w = drawText(ctx, s, x: curX, y: y, size: size, weight: .mathItalic, color: c ?? color)
            curX += w + 1.5
        case .b(let s, let c):
            let w = drawText(ctx, s, x: curX, y: y, size: size, weight: .mathBold, color: c ?? color)
            curX += w + 1.5
        case .r(let s, let c):
            let w = drawText(ctx, s, x: curX, y: y, size: size, weight: .mathRoman, color: c ?? color)
            curX += w + 1.5
        case .op(let s, let c):
            let w = drawText(ctx, s, x: curX, y: y + size * 0.03, size: size * 0.90, weight: .medium, color: c ?? color)
            curX += w + 2.0
        case .sub(let s, let c):
            let w = drawText(ctx, s, x: curX, y: y - size * 0.25, size: size * 0.73, weight: .mathRoman, color: c ?? color)
            curX += w + 1.5
        case .sup(let s, let c):
            let w = drawText(ctx, s, x: curX, y: y + size * 0.38, size: size * 0.73, weight: .mathRoman, color: c ?? color)
            curX += w + 1.5
        case .subSup(let subS, let supS, let c):
            let wSub = drawText(ctx, subS, x: curX, y: y - size * 0.28, size: size * 0.71, weight: .mathRoman, color: c ?? color)
            let wSup = drawText(ctx, supS, x: curX, y: y + size * 0.42, size: size * 0.71, weight: .mathRoman, color: c ?? color)
            curX += max(wSub, wSup) + 2.5
        case .frac(let num, let den, let c):
            let subSize = size * 0.85
            let wNum = measureMathTokens(num, baseSize: subSize)
            let wDen = measureMathTokens(den, baseSize: subSize)
            let barW = max(wNum, wDen) + 8.0
            let midX = curX + barW / 2.0 + 2.0
            let barY = y + size * 0.30
            let numHasSub = num.contains {
                if case .sub = $0 { return true }
                if case .subSup = $0 { return true }
                return false
            }
            let denHasSup = den.contains {
                if case .sup = $0 { return true }
                if case .subSup = $0 { return true }
                return false
            }
            let numY = barY + size * (numHasSub ? 0.48 : 0.30)
            let denY = barY - size * (denHasSup ? 0.98 : 0.78)
            drawMath(ctx, num, x: midX, y: numY, size: subSize, color: c ?? color, align: .center)
            drawMath(ctx, den, x: midX, y: denY, size: subSize, color: c ?? color, align: .center)
            ctx.saveGState()
            ctx.setStrokeColor(c ?? color)
            ctx.setLineWidth(max(1.4, size * 0.065))
            ctx.move(to: CGPoint(x: curX + 2.0, y: barY))
            ctx.addLine(to: CGPoint(x: curX + 2.0 + barW, y: barY))
            ctx.strokePath()
            ctx.restoreGState()
            curX += barW + 4.0
        case .sqrt(let inner, let c):
            let subSize = size * 0.88
            let wIn = measureMathTokens(inner, baseSize: subSize)
            let hasFrac = inner.contains { if case .frac = $0 { return true } else { return false } }
            let radW = size * (hasFrac ? 0.68 : 0.55)
            let topY = y + size * (hasFrac ? 1.18 : 0.82)
            let botY = y + size * (hasFrac ? -0.52 : 0.02)
            let hookY = y + size * (hasFrac ? 0.05 : 0.32)
            let col = c ?? color
            ctx.saveGState()
            ctx.setStrokeColor(col)
            ctx.setLineWidth(max(1.4, size * 0.065))
            ctx.setLineCap(.round)
            ctx.setLineJoin(.round)
            ctx.move(to: CGPoint(x: curX + 1, y: hookY))
            ctx.addLine(to: CGPoint(x: curX + radW * 0.38, y: botY))
            ctx.addLine(to: CGPoint(x: curX + radW * 0.88, y: topY))
            ctx.addLine(to: CGPoint(x: curX + radW + wIn + 4, y: topY))
            ctx.strokePath()
            ctx.restoreGState()
            drawMath(ctx, inner, x: curX + radW + 1, y: y, size: subSize, color: col, align: .left)
            curX += radW + wIn + 8.0
        case .space(let sp):
            curX += sp
        }
    }
    return totalW
}

// ----------------------------------------------------------------------------
// Visual Primitives: Gemini 4 Argon Glows, Prism Beams, Divided Cards, Icons
// ----------------------------------------------------------------------------
let cArgonCobalt = rgba(18, 78, 235, 1.0)    // Deep Gemini 4 Argon Cobalt (#124EEB)
let cArgonElectric = rgba(59, 140, 255, 1.0) // Electric Argon Blue (#3B8CFF)
let cArgonIce = rgba(124, 194, 255, 1.0)     // Icy Argon Cyan-Sky (#7CC2FF)

func drawRadialGlow(
    _ ctx: CGContext,
    cx: CGFloat,
    cy: CGFloat,
    radius: CGFloat,
    color: CGColor,
    alpha: CGFloat
) {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let colors = [withAlpha(color, alpha), withAlpha(color, alpha * 0.38), withAlpha(color, 0.0)] as CFArray
    let locations: [CGFloat] = [0.0, 0.46, 1.0]
    guard let grad = CGGradient(colorsSpace: colorSpace, colors: colors, locations: locations) else { return }
    ctx.saveGState()
    ctx.drawRadialGradient(
        grad,
        startCenter: CGPoint(x: cx, y: cy),
        startRadius: 0,
        endCenter: CGPoint(x: cx, y: cy),
        endRadius: radius,
        options: .drawsAfterEndLocation
    )
    ctx.restoreGState()
}

func drawArgonStippleMatrix(
    _ ctx: CGContext,
    cx: CGFloat,
    cy: CGFloat,
    rx: CGFloat,
    ry: CGFloat,
    color: CGColor,
    t: CGFloat,
    maxAlpha: CGFloat = 0.14
) {
    let spacing: CGFloat = 24.0
    let startX = max(20, cx - rx)
    let endX   = min(W - 20, cx + rx)
    let startY = max(20, cy - ry)
    let endY   = min(H - 20, cy + ry)
    let phase = t * 2.0 * .pi

    ctx.saveGState()
    var gx = floor(startX / spacing) * spacing
    while gx <= endX {
        var gy = floor(startY / spacing) * spacing
        while gy <= endY {
            let dx = (gx - cx) / rx
            let dy = (gy - cy) / ry
            let dist2 = dx * dx + dy * dy
            if dist2 < 1.0 {
                let falloff = (1.0 - dist2) * (1.0 - dist2)
                let wave = 0.55 + 0.45 * sin(phase + gx * 0.018 + gy * 0.022)
                let a = maxAlpha * falloff * wave
                if a > 0.018 {
                    ctx.setFillColor(withAlpha(color, a))
                    ctx.fillEllipse(in: CGRect(x: gx - 1.3, y: gy - 1.3, width: 2.6, height: 2.6))
                }
            }
            gy += spacing
        }
        gx += spacing
    }
    ctx.restoreGState()
}

func drawStageCornerGlows(
    _ ctx: CGContext,
    topLeftColor: CGColor = cBlue,
    bottomRightColor: CGColor = cViolet,
    t: CGFloat,
    intensity: CGFloat = 0.16
) {
    let pulse = sin(t * 2.0 * .pi)
    let r1: CGFloat = 780 + 35 * pulse
    let r2: CGFloat = 740 - 30 * pulse
    drawRadialGlow(ctx, cx: 110, cy: 1010, radius: r1 * 1.08, color: cArgonCobalt, alpha: intensity * 0.72)
    drawRadialGlow(ctx, cx: 80, cy: 1020, radius: r1 * 0.82, color: topLeftColor, alpha: intensity + 0.02 * pulse)
    drawRadialGlow(ctx, cx: 1840, cy: 60, radius: r2, color: bottomRightColor, alpha: intensity - 0.02 * pulse)

    drawArgonStippleMatrix(ctx, cx: 170, cy: 960, rx: 380, ry: 210, color: cArgonIce, t: t, maxAlpha: 0.13)
    drawArgonStippleMatrix(ctx, cx: 1760, cy: 110, rx: 340, ry: 180, color: bottomRightColor, t: t + 0.35, maxAlpha: 0.11)
}

func roundedPath(_ rect: CGRect, radius: CGFloat) -> CGPath {
    return CGPath(roundedRect: rect, cornerWidth: radius, cornerHeight: radius, transform: nil)
}

func drawCard(
    _ ctx: CGContext,
    x: CGFloat,
    y: CGFloat,
    w: CGFloat,
    h: CGFloat,
    radius: CGFloat = 24,
    fill: CGColor = cSurface,
    stroke: CGColor? = cStroke,
    lineWidth: CGFloat = 1.4
) {
    let rect = CGRect(x: x, y: y, width: w, height: h)
    let path = roundedPath(rect, radius: radius)
    ctx.saveGState()
    ctx.addPath(path)
    ctx.setFillColor(fill)
    ctx.fillPath()
    if let s = stroke {
        ctx.addPath(path)
        ctx.setStrokeColor(s)
        ctx.setLineWidth(lineWidth)
        ctx.strokePath()
    }
    ctx.restoreGState()
}

func drawDivider(_ ctx: CGContext, x1: CGFloat, y1: CGFloat, x2: CGFloat, y2: CGFloat, color: CGColor = cStroke, width: CGFloat = 1.2, dashed: Bool = false) {
    ctx.saveGState()
    ctx.setStrokeColor(color)
    ctx.setLineWidth(width)
    if dashed { ctx.setLineDash(phase: 0, lengths: [6, 5]) }
    ctx.move(to: CGPoint(x: x1, y: y1))
    ctx.addLine(to: CGPoint(x: x2, y: y2))
    ctx.strokePath()
    ctx.restoreGState()
}

func drawKeynoteCard(
    _ ctx: CGContext,
    x: CGFloat,
    y: CGFloat,
    w: CGFloat,
    h: CGFloat,
    radius: CGFloat = 32,
    accent: CGColor = cBlue,
    headerHeight: CGFloat? = nil,
    focus: CGFloat = 0.5
) {
    let rect = CGRect(x: x, y: y, width: w, height: h)
    let path = roundedPath(rect, radius: radius)

    ctx.saveGState()
    ctx.addPath(path)
    ctx.setFillColor(cCardBg)
    ctx.fillPath()

    ctx.addPath(path)
    ctx.clip()
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let topAlpha: CGFloat = 0.09 + 0.05 * focus
    let gradColors = [withAlpha(accent, topAlpha), withAlpha(cArgonCobalt, topAlpha * 0.45), withAlpha(accent, 0.0)] as CFArray
    let gradLocs: [CGFloat] = [0.0, 0.45, 1.0]
    if let grad = CGGradient(colorsSpace: colorSpace, colors: gradColors, locations: gradLocs) {
        let washBottom = headerHeight != nil ? (y + h - headerHeight! - 45) : (y + h * 0.42)
        ctx.drawLinearGradient(
            grad,
            start: CGPoint(x: x + w / 2, y: y + h),
            end: CGPoint(x: x + w / 2, y: washBottom),
            options: []
        )
    }
    ctx.restoreGState()

    let borderAlpha: CGFloat = 0.52 + 0.36 * focus
    ctx.saveGState()
    ctx.addPath(path)
    ctx.setStrokeColor(withAlpha(accent, borderAlpha))
    ctx.setLineWidth(1.6)
    ctx.strokePath()
    ctx.restoreGState()

    if let hh = headerHeight {
        let divY = y + h - hh
        drawDivider(ctx, x1: x, y1: divY, x2: x + w, y2: divY, color: withAlpha(accent, borderAlpha * 0.85), width: 1.2)
    }
}

enum KeynoteIcon {
    case sparkle, layers, bolt, check, cross, target, sliders, shield, branch
}

func drawKeynoteIcon(_ ctx: CGContext, _ icon: KeynoteIcon, cx: CGFloat, cy: CGFloat, size: CGFloat = 8.5, color: CGColor = cInk) {
    ctx.saveGState()
    ctx.setStrokeColor(color)
    ctx.setFillColor(color)
    ctx.setLineWidth(1.7)
    ctx.setLineCap(.round)
    ctx.setLineJoin(.round)

    switch icon {
    case .sparkle:
        let p = CGMutablePath()
        p.move(to: CGPoint(x: cx, y: cy + size))
        p.addQuadCurve(to: CGPoint(x: cx + size, y: cy), control: CGPoint(x: cx + size * 0.18, y: cy + size * 0.18))
        p.addQuadCurve(to: CGPoint(x: cx, y: cy - size), control: CGPoint(x: cx + size * 0.18, y: cy - size * 0.18))
        p.addQuadCurve(to: CGPoint(x: cx - size, y: cy), control: CGPoint(x: cx - size * 0.18, y: cy - size * 0.18))
        p.addQuadCurve(to: CGPoint(x: cx, y: cy + size), control: CGPoint(x: cx - size * 0.18, y: cy + size * 0.18))
        p.closeSubpath()
        ctx.addPath(p)
        ctx.fillPath()
    case .layers:
        for dy in [size * 0.30, -size * 0.30] {
            let p = CGMutablePath()
            p.move(to: CGPoint(x: cx, y: cy + dy + size * 0.52))
            p.addLine(to: CGPoint(x: cx + size, y: cy + dy))
            p.addLine(to: CGPoint(x: cx, y: cy + dy - size * 0.52))
            p.addLine(to: CGPoint(x: cx - size, y: cy + dy))
            p.closeSubpath()
            ctx.addPath(p)
            ctx.strokePath()
        }
    case .bolt:
        let p = CGMutablePath()
        p.move(to: CGPoint(x: cx + size * 0.15, y: cy + size))
        p.addLine(to: CGPoint(x: cx - size * 0.65, y: cy + size * 0.05))
        p.addLine(to: CGPoint(x: cx + size * 0.10, y: cy + size * 0.05))
        p.addLine(to: CGPoint(x: cx - size * 0.15, y: cy - size))
        p.addLine(to: CGPoint(x: cx + size * 0.65, y: cy - size * 0.05))
        p.addLine(to: CGPoint(x: cx - size * 0.10, y: cy - size * 0.05))
        p.closeSubpath()
        ctx.addPath(p)
        ctx.fillPath()
    case .check:
        ctx.strokeEllipse(in: CGRect(x: cx - size, y: cy - size, width: size * 2, height: size * 2))
        ctx.move(to: CGPoint(x: cx - size * 0.45, y: cy - size * 0.05))
        ctx.addLine(to: CGPoint(x: cx - size * 0.10, y: cy - size * 0.42))
        ctx.addLine(to: CGPoint(x: cx + size * 0.52, y: cy + size * 0.38))
        ctx.strokePath()
    case .cross:
        ctx.strokeEllipse(in: CGRect(x: cx - size, y: cy - size, width: size * 2, height: size * 2))
        let d = size * 0.42
        ctx.move(to: CGPoint(x: cx - d, y: cy - d))
        ctx.addLine(to: CGPoint(x: cx + d, y: cy + d))
        ctx.move(to: CGPoint(x: cx - d, y: cy + d))
        ctx.addLine(to: CGPoint(x: cx + d, y: cy - d))
        ctx.strokePath()
    case .target:
        ctx.strokeEllipse(in: CGRect(x: cx - size, y: cy - size, width: size * 2, height: size * 2))
        ctx.fillEllipse(in: CGRect(x: cx - size * 0.36, y: cy - size * 0.36, width: size * 0.72, height: size * 0.72))
    case .sliders:
        for (idx, dy) in [size * 0.48, -size * 0.48].enumerated() {
            ctx.move(to: CGPoint(x: cx - size, y: cy + dy))
            ctx.addLine(to: CGPoint(x: cx + size, y: cy + dy))
            ctx.strokePath()
            let kx = idx == 0 ? cx - size * 0.3 : cx + size * 0.35
            ctx.fillEllipse(in: CGRect(x: kx - 3.2, y: cy + dy - 3.2, width: 6.4, height: 6.4))
        }
    case .shield:
        let p = CGMutablePath()
        p.move(to: CGPoint(x: cx, y: cy + size))
        p.addLine(to: CGPoint(x: cx + size * 0.85, y: cy + size * 0.55))
        p.addLine(to: CGPoint(x: cx + size * 0.70, y: cy - size * 0.35))
        p.addLine(to: CGPoint(x: cx, y: cy - size))
        p.addLine(to: CGPoint(x: cx - size * 0.70, y: cy - size * 0.35))
        p.addLine(to: CGPoint(x: cx - size * 0.85, y: cy + size * 0.55))
        p.closeSubpath()
        ctx.addPath(p)
        ctx.strokePath()
    case .branch:
        ctx.move(to: CGPoint(x: cx - size * 0.6, y: cy - size * 0.7))
        ctx.addLine(to: CGPoint(x: cx - size * 0.6, y: cy + size * 0.7))
        ctx.move(to: CGPoint(x: cx - size * 0.6, y: cy))
        ctx.addLine(to: CGPoint(x: cx + size * 0.6, y: cy + size * 0.5))
        ctx.strokePath()
        ctx.fillEllipse(in: CGRect(x: cx - size * 0.6 - 3, y: cy + size * 0.7 - 3, width: 6, height: 6))
        ctx.fillEllipse(in: CGRect(x: cx + size * 0.6 - 3, y: cy + size * 0.5 - 3, width: 6, height: 6))
    }
    ctx.restoreGState()
}

func drawFeatureRow(
    _ ctx: CGContext,
    icon: KeynoteIcon? = .sparkle,
    iconColor: CGColor = cInk,
    spans: [(String, FontWeight, CGColor)],
    x: CGFloat,
    y: CGFloat,
    size: CGFloat = 18.5
) {
    var curX = x
    if let ic = icon {
        drawKeynoteIcon(ctx, ic, cx: x + 9, cy: y + size * 0.35, size: 8.8, color: iconColor)
        curX += 30
    }
    for (txt, w, col) in spans {
        let adv = drawText(ctx, txt, x: curX, y: y, size: size, weight: w, color: col)
        curX += adv
    }
}

func drawVectorArrow(
    _ ctx: CGContext,
    x1: CGFloat,
    y1: CGFloat,
    x2: CGFloat,
    y2: CGFloat,
    color: CGColor,
    width: CGFloat = 2.6,
    headLength: CGFloat = 12,
    headAngle: CGFloat = 0.45
) {
    let dx = x2 - x1
    let dy = y2 - y1
    let len = hypot(dx, dy)
    guard len > 1 else { return }
    let angle = atan2(dy, dx)
    let p1 = CGPoint(x: x2 - headLength * cos(angle - headAngle), y: y2 - headLength * sin(angle - headAngle))
    let p2 = CGPoint(x: x2 - headLength * cos(angle + headAngle), y: y2 - headLength * sin(angle + headAngle))

    ctx.saveGState()
    ctx.setStrokeColor(color)
    ctx.setLineWidth(width)
    ctx.setLineCap(.round)
    ctx.move(to: CGPoint(x: x1, y: y1))
    ctx.addLine(to: CGPoint(x: x2 - headLength * 0.5 * cos(angle), y: y2 - headLength * 0.5 * sin(angle)))
    ctx.strokePath()

    ctx.setFillColor(color)
    ctx.move(to: CGPoint(x: x2, y: y2))
    ctx.addLine(to: p1)
    ctx.addLine(to: p2)
    ctx.closePath()
    ctx.fillPath()
    ctx.restoreGState()
}

func drawFlowingLine(
    _ ctx: CGContext,
    x1: CGFloat,
    y1: CGFloat,
    x2: CGFloat,
    y2: CGFloat,
    color: CGColor,
    t: CGFloat,
    dots: Int = 1
) {
    drawDivider(ctx, x1: x1, y1: y1, x2: x2, y2: y2, color: withAlpha(color, 0.32), width: 2.0)
    for i in 0..<dots {
        let phase = (t + (CGFloat(i) + 0.5) / CGFloat(dots)).truncatingRemainder(dividingBy: 1.0)
        let px = x1 + (x2 - x1) * phase
        let py = y1 + (y2 - y1) * phase
        ctx.saveGState()
        ctx.setFillColor(withAlpha(color, 0.25))
        ctx.fillEllipse(in: CGRect(x: px - 6.5, y: py - 6.5, width: 13, height: 13))
        ctx.setFillColor(color)
        ctx.fillEllipse(in: CGRect(x: px - 3.2, y: py - 3.2, width: 6.4, height: 6.4))
        ctx.restoreGState()
    }
}

func isoPoint(cx: CGFloat, cy: CGFloat, u: CGFloat, v: CGFloat, rx: CGFloat = 290, ry: CGFloat = 84) -> CGPoint {
    return CGPoint(
        x: cx + (u - v) * 0.5 * rx,
        y: cy + (u + v) * 0.5 * ry
    )
}

func drawIsoPlane(
    _ ctx: CGContext,
    cx: CGFloat,
    cy: CGFloat,
    rx: CGFloat = 290,
    ry: CGFloat = 84,
    fill: CGColor,
    stroke: CGColor,
    lineWidth: CGFloat = 1.8
) {
    let p0 = isoPoint(cx: cx, cy: cy, u: -1, v: -1, rx: rx, ry: ry)
    let p1 = isoPoint(cx: cx, cy: cy, u:  1, v: -1, rx: rx, ry: ry)
    let p2 = isoPoint(cx: cx, cy: cy, u:  1, v:  1, rx: rx, ry: ry)
    let p3 = isoPoint(cx: cx, cy: cy, u: -1, v:  1, rx: rx, ry: ry)

    let path = CGMutablePath()
    path.move(to: p0)
    path.addLine(to: p1)
    path.addLine(to: p2)
    path.addLine(to: p3)
    path.closeSubpath()

    ctx.saveGState()
    ctx.addPath(path)
    ctx.setFillColor(fill)
    ctx.fillPath()
    ctx.addPath(path)
    ctx.setStrokeColor(stroke)
    ctx.setLineWidth(lineWidth)
    ctx.strokePath()
    ctx.restoreGState()
}

func columnFocus(index: Int, count: Int, t: CGFloat) -> CGFloat {
    let center = (CGFloat(index) + 0.5) / CGFloat(count)
    var dist = abs(t - center)
    if dist > 0.5 { dist = 1.0 - dist }
    let sigma: CGFloat = 0.25
    return exp(-(dist * dist) / (2 * sigma * sigma))
}

// ============================================================================
// MAIN SLIDE RENDERER (Slides 1..15)
// Supports arbitrary resolution scale (e.g. 1.0 for 1080p, 2.0 for 4K Retina)
// ============================================================================
func renderFrame(slideIndex: Int, t: CGFloat, scale: CGFloat = 1.0) -> CGImage {
    let pw = Int(W * scale)
    let ph = Int(H * scale)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let ctx = CGContext(
        data: nil,
        width: pw,
        height: ph,
        bitsPerComponent: 8,
        bytesPerRow: pw * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )!

    ctx.scaleBy(x: scale, y: scale)

    // 1. Pure Pitch-Black #000000 Canvas
    ctx.setFillColor(cBg)
    ctx.fill(CGRect(x: 0, y: 0, width: W, height: H))

    let angle = t * 2.0 * .pi

    switch slideIndex {

    // ------------------------------------------------------------------------
    // SLIDE 01: GEMINI 4 ARGON DARK STAGE COVER (15.0s MAJESTIC KINETIC CYCLE)
    // ------------------------------------------------------------------------
    case 1:
        drawStageCornerGlows(ctx, topLeftColor: cArgonElectric, bottomRightColor: cViolet, t: t, intensity: 0.24)

        let marqueeTop = "CLASSIFY 100 CLAUSE TYPES   ·   RETRIEVE PRECEDENTS   ·   AUDIT FAILURE MODES"
        let marqueeBot = "DISTILBERT + LORA   ·   SENTENCE TRANSFORMERS   ·   TEMPERATURE CALIBRATION"
        let drift = 46.0 * sin(angle)
        drawText(ctx, marqueeTop, x: 960 + drift, y: 785, size: 122, weight: .bold, color: withAlpha(cInk, 0.048), align: .center, tracking: -2.8)
        drawText(ctx, marqueeBot, x: 960 - drift, y: 165, size: 122, weight: .bold, color: withAlpha(cArgonElectric, 0.055), align: .center, tracking: -2.8)

        drawTwoToneTitle(ctx, leftText: "ClauseGuard ", rightText: "AI", rightColor: cArgonElectric, y: 560, size: 112, centerX: 960, tracking: -2.8)

        let clipRect = CGRect(x: 160, y: 425, width: 1600, height: 80)
        ctx.saveGState()
        ctx.clip(to: clipRect)

        let phrases: [(verb: String, rest: String, col: CGColor)] = [
            ("Classify ", "every contract provision across 100 SEC checklist types (0.827 Macro-F1)", cArgonElectric),
            ("Retrieve ", "precedents across 60,000 clauses via BGE-small, E5 & TF-IDF geometry", cCyan),
            ("Audit ", "ECE calibration (T=2.47), contract-kind bias (leases 75.8% vs loans 89.4%) & red lines", cAmber),
            ("Classify ", "every contract provision across 100 SEC checklist types (0.827 Macro-F1)", cArgonElectric)
        ]

        let scaled = t * 3.0
        let seg = Int(floor(scaled))
        let frac = scaled - CGFloat(seg)
        let glide: CGFloat
        if frac < 0.85 {
            glide = 0.0
        } else {
            let u = (frac - 0.85) / 0.15
            glide = u * u * (3.0 - 2.0 * u)
        }
        let scrollPos = CGFloat(seg) + glide
        let slotH: CGFloat = 80.0

        for (idx, item) in phrases.enumerated() {
            let itemY = 448.0 + (CGFloat(idx) - scrollPos) * slotH
            if itemY > 345 && itemY < 545 {
                let dist = abs(CGFloat(idx) - scrollPos)
                let alpha = max(0.0, 1.0 - dist * 1.15)
                let w1 = measureText(item.verb, size: 38, weight: .bold, tracking: -0.8).width
                let w2 = measureText(item.rest, size: 38, weight: .medium, tracking: -0.6).width
                let sx = 960 - (w1 + w2) / 2.0
                drawText(ctx, item.verb, x: sx, y: itemY, size: 38, weight: .bold, color: withAlpha(item.col, alpha), tracking: -0.8)
                drawText(ctx, item.rest, x: sx + w1, y: itemY, size: 38, weight: .medium, color: withAlpha(cInk, alpha), tracking: -0.6)
            }
        }
        ctx.restoreGState()

        drawText(ctx, "Jonathan   ·   Keira   ·   Alexis", x: 960, y: 300, size: 28, weight: .bold, color: cInk, align: .center, tracking: 1.5)

    // ------------------------------------------------------------------------
    // SLIDE 02: THE LEGAL DUE-DILIGENCE BOTTLENECK (15.0s LIVE COMPARISON CYCLE)
    // ------------------------------------------------------------------------
    case 2:
        drawStageCornerGlows(ctx, topLeftColor: cCoral, bottomRightColor: cArgonElectric, t: t, intensity: 0.18)
        drawTwoToneTitle(ctx, leftText: "Thousands of hours ", rightText: "reading contracts", rightColor: cCoral)
        drawText(ctx, "M&A due diligence, financings, and supplier audits bottleneck legal teams on repetitive clause review", x: 960, y: 882, size: 24, weight: .medium, color: cSlate, align: .center)

        drawText(ctx, "80k clauses", x: 105, y: 675, size: 96, weight: .bold, color: cCoral, tracking: -3.0)
        drawText(ctx, "across U.S. SEC Exhibit-10 filings requiring", x: 105, y: 620, size: 24, weight: .medium, color: cInk)
        drawText(ctx, "consistent 100-type tagging & precedent search.", x: 105, y: 584, size: 24, weight: .medium, color: cSlate)

        let barProgress = 0.25 + 0.75 * (0.5 - 0.5 * cos(angle))
        drawText(ctx, "Manual Review (hours / SEC filing)", x: 105, y: 540, size: 17.5, weight: .bold, color: cCoral)
        drawText(ctx, "ClauseGuard AI (1.9 ms / clause on A100)", x: 745, y: 540, size: 17.5, weight: .bold, color: cArgonIce, align: .right)
        drawCard(ctx, x: 105, y: 512, w: 640, h: 16, radius: 8, fill: rgba(255, 255, 255, 0.09), stroke: nil)
        drawCard(ctx, x: 105, y: 512, w: 640 * barProgress, h: 16, radius: 8, fill: withAlpha(cCoral, 0.88), stroke: nil)
        drawCard(ctx, x: 105, y: 512, w: 42, h: 16, radius: 8, fill: cArgonIce, stroke: nil)

        let manualSteps: [(icon: KeynoteIcon, title: String, sub: String)] = [
            (.layers, "Read every paragraph line-by-line", "200+ pages per filing across hundreds of agreements"),
            (.sliders, "Tag checklist categories by hand", "Inconsistent Governing Law, Venue & Indemnity tags"),
            (.target, "Hunt for precedents in folders", "Keyword Ctrl+F misses equivalent legal phrasings")
        ]
        for (idx, ms) in manualSteps.enumerated() {
            let sy: CGFloat = 440 - CGFloat(idx) * 106
            let stepFocus = columnFocus(index: idx, count: 3, t: t)
            if stepFocus > 0.55 {
                drawRadialGlow(ctx, cx: 120, cy: sy + 10, radius: 44, color: cCoral, alpha: (stepFocus - 0.55) * 0.45)
            }
            drawKeynoteIcon(ctx, ms.icon, cx: 120, cy: sy + 10, size: 11.5, color: stepFocus > 0.55 ? cInk : cCoral)
            drawText(ctx, ms.title, x: 150, y: sy + 2, size: 22.5, weight: .bold, color: cInk)
            drawText(ctx, ms.sub, x: 150, y: sy - 27, size: 18.5, weight: .medium, color: cSlate)
        }

        let f0 = columnFocus(index: 0, count: 2, t: t)
        drawKeynoteCard(ctx, x: 790, y: 175, w: 505, h: 635, radius: 34, accent: cArgonElectric, headerHeight: 140, focus: f0)
        drawText(ctx, "NLP Core Capabilities", x: 1042.5, y: 742, size: 28, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "Classify & Retrieve in Real Time", x: 1042.5, y: 700, size: 20, weight: .medium, color: cArgonIce, align: .center)

        drawText(ctx, "Accelerate junior lawyer review across", x: 826, y: 616, size: 20, weight: .bold, color: cInk)
        drawText(ctx, "every commercial contract:", x: 826, y: 588, size: 20, weight: .bold, color: cInk)

        drawFeatureRow(ctx, icon: .sparkle, iconColor: cArgonElectric, spans: [
            ("Classify into ", .medium, cSlate), ("100 SEC clause types (88.0% Acc)", .bold, cArgonIce)
        ], x: 826, y: 522, size: 18.5)
        drawFeatureRow(ctx, icon: .layers, iconColor: cInk, spans: [
            ("Compare ", .medium, cSlate), ("TF-IDF (0.829), BiLSTM & DistilBERT", .bold, cInk)
        ], x: 826, y: 464, size: 18.5)
        drawFeatureRow(ctx, icon: .bolt, iconColor: cCyan, spans: [
            ("Audit ", .medium, cSlate), ("13 retrieval encodings & 78 variants", .bold, cCyan)
        ], x: 826, y: 406, size: 18.5)
        drawFeatureRow(ctx, icon: .target, iconColor: cInk, spans: [
            ("Answer lawyer queries (", .medium, cSlate), ("86-88% constraint hit", .bold, cArgonIce), (")", .medium, cSlate)
        ], x: 826, y: 348, size: 18.5)
        drawFeatureRow(ctx, icon: .branch, iconColor: cInk, spans: [
            ("Cut trainable weights by ", .medium, cSlate), ("98.8% with LoRA", .bold, cGreen)
        ], x: 826, y: 290, size: 18.5)

        drawText(ctx, "*Trained on 60,000 U.S. SEC legal provisions", x: 1042.5, y: 208, size: 15, weight: .regular, color: cMuted, align: .center)

        let f1 = columnFocus(index: 1, count: 2, t: t)
        drawKeynoteCard(ctx, x: 1325, y: 175, w: 505, h: 635, radius: 34, accent: cGreen, headerHeight: 140, focus: f1)
        drawText(ctx, "European Audit Standard", x: 1577.5, y: 742, size: 28, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "No Black Box in Production", x: 1577.5, y: 700, size: 20, weight: .medium, color: cGreen, align: .center)

        drawText(ctx, "Every model choice is defended with", x: 1361, y: 616, size: 20, weight: .bold, color: cInk)
        drawText(ctx, "pre-registered rules and bias slices:", x: 1361, y: 588, size: 20, weight: .bold, color: cInk)

        drawFeatureRow(ctx, icon: .check, iconColor: cGreen, spans: [
            ("Official ", .medium, cSlate), ("60k / 10k / 10k", .bold, cGreen), (" LexGLUE split", .medium, cSlate)
        ], x: 1361, y: 522, size: 18.5)
        drawFeatureRow(ctx, icon: .sliders, iconColor: cInk, spans: [
            ("Seeds ", .medium, cSlate), ("{0, 1, 2}", .bold, cInk), (" · McNemar & 1k-bootstrap CI", .medium, cSlate)
        ], x: 1361, y: 464, size: 18.5)
        drawFeatureRow(ctx, icon: .shield, iconColor: cAmber, spans: [
            ("Audit ", .medium, cSlate), ("138× Zipfian tail", .bold, cAmber), (" (0.96 vs 0.76 F1)", .medium, cSlate)
        ], x: 1361, y: 406, size: 18.5)
        drawFeatureRow(ctx, icon: .sparkle, iconColor: cCyan, spans: [
            ("Slice ", .medium, cSlate), ("contract kinds, parties & jurisdictions", .bold, cCyan)
        ], x: 1361, y: 348, size: 18.5)
        drawFeatureRow(ctx, icon: .cross, iconColor: cCoral, spans: [
            ("Flag ", .medium, cSlate), ("leases (75.8% Acc) & ALL-CAPS (0.52 F1)", .bold, cCoral)
        ], x: 1361, y: 290, size: 18.5)

        drawText(ctx, "*Pre-registered protocol in README & DECISIONS.md", x: 1577.5, y: 208, size: 15, weight: .regular, color: cMuted, align: .center)

    // ------------------------------------------------------------------------
    // SLIDE 03: END-TO-END ARCHITECTURE BLUEPRINT (GOOGLE CLOUD NEXT ISOMETRIC STACK)
    // ------------------------------------------------------------------------
    case 3:
        drawStageCornerGlows(ctx, topLeftColor: cBlue, bottomRightColor: cViolet, t: t, intensity: 0.20)
        drawTwoToneTitle(ctx, leftText: "ClauseGuard AI ", rightText: "Architecture Blueprint", rightColor: cBlue)
        drawText(ctx, "Five reproducible layers from raw SEC filings to paired statistical evaluation and bias slices", x: 960, y: 882, size: 24, weight: .medium, color: cSlate, align: .center)

        let isoX: CGFloat = 530
        let layerY: [CGFloat] = [200, 330, 460, 590, 720]
        let rx: CGFloat = 290
        let ry: CGFloat = 84

        let activeWave = t * 5.0
        func planeFocus(_ idx: Int) -> CGFloat {
            let d1 = abs(CGFloat(idx) - activeWave)
            let d2 = abs(CGFloat(idx) + 5.0 - activeWave)
            let d3 = abs(CGFloat(idx) - 5.0 - activeWave)
            let d = min(d1, min(d2, d3))
            return max(0.0, 1.0 - d * 1.1)
        }

        for u in stride(from: -0.6, through: 0.6, by: 0.6) {
            let pBot = isoPoint(cx: isoX, cy: layerY[0], u: CGFloat(u), v: 0, rx: rx, ry: ry)
            let pTop = isoPoint(cx: isoX, cy: layerY[4], u: CGFloat(u), v: 0, rx: rx, ry: ry)
            drawFlowingLine(ctx, x1: pBot.x, y1: pBot.y, x2: pTop.x, y2: pTop.y, color: cBlue, t: t * 2.0, dots: 3)
        }

        let f0 = planeFocus(0)
        drawIsoPlane(ctx, cx: isoX, cy: layerY[0], rx: rx, ry: ry, fill: rgba(35, 60, 130, 0.38 + 0.22 * f0), stroke: cBlue, lineWidth: 1.8 + 1.4 * f0)
        for u in stride(from: -0.75, through: 0.75, by: 0.3) {
            for v in stride(from: -0.75, through: 0.75, by: 0.3) {
                let pt = isoPoint(cx: isoX, cy: layerY[0], u: CGFloat(u), v: CGFloat(v), rx: rx, ry: ry)
                let wave = 0.5 + 0.5 * sin(angle * 2.0 - CGFloat(u + v) * 2.2)
                ctx.saveGState()
                ctx.setFillColor(withAlpha(cBlue, 0.55 + 0.45 * wave))
                ctx.fillEllipse(in: CGRect(x: pt.x - 3.5, y: pt.y - 2.5, width: 7, height: 5))
                ctx.restoreGState()
            }
        }

        let f1 = planeFocus(1)
        drawIsoPlane(ctx, cx: isoX, cy: layerY[1], rx: rx, ry: ry, fill: rgba(18, 45, 58, 0.44 + 0.22 * f1), stroke: cCyan, lineWidth: 1.8 + 1.4 * f1)
        for g in stride(from: -0.6, through: 0.6, by: 0.4) {
            let a1 = isoPoint(cx: isoX, cy: layerY[1], u: CGFloat(g), v: -0.8, rx: rx, ry: ry)
            let a2 = isoPoint(cx: isoX, cy: layerY[1], u: CGFloat(g), v:  0.8, rx: rx, ry: ry)
            let b1 = isoPoint(cx: isoX, cy: layerY[1], u: -0.8, v: CGFloat(g), rx: rx, ry: ry)
            let b2 = isoPoint(cx: isoX, cy: layerY[1], u:  0.8, v: CGFloat(g), rx: rx, ry: ry)
            drawDivider(ctx, x1: a1.x, y1: a1.y, x2: a2.x, y2: a2.y, color: withAlpha(cCyan, 0.40 + 0.30 * f1), width: 1.3)
            drawDivider(ctx, x1: b1.x, y1: b1.y, x2: b2.x, y2: b2.y, color: withAlpha(cCyan, 0.40 + 0.30 * f1), width: 1.3)
        }
        let scanU = -0.65 + 1.30 * (0.5 - 0.5 * cos(angle * 2.0))
        let sA = isoPoint(cx: isoX, cy: layerY[1], u: scanU, v: -0.82, rx: rx, ry: ry)
        let sB = isoPoint(cx: isoX, cy: layerY[1], u: scanU, v:  0.82, rx: rx, ry: ry)
        drawDivider(ctx, x1: sA.x, y1: sA.y, x2: sB.x, y2: sB.y, color: cCyan, width: 2.6)

        let f2 = planeFocus(2)
        drawIsoPlane(ctx, cx: isoX, cy: layerY[2], rx: rx, ry: ry, fill: rgba(42, 22, 68, 0.46 + 0.24 * f2), stroke: cViolet, lineWidth: 2.0 + 1.4 * f2)
        let meshNodes: [(CGFloat, CGFloat)] = [(-0.6, -0.5), (0.0, -0.6), (0.6, -0.3), (-0.4, 0.3), (0.2, 0.4), (0.7, 0.5), (0.0, 0.0)]
        for i in 0..<meshNodes.count {
            for j in (i+1)..<meshNodes.count {
                if (i + j) % 2 == 0 {
                    let pA = isoPoint(cx: isoX, cy: layerY[2], u: meshNodes[i].0, v: meshNodes[i].1, rx: rx, ry: ry)
                    let pB = isoPoint(cx: isoX, cy: layerY[2], u: meshNodes[j].0, v: meshNodes[j].1, rx: rx, ry: ry)
                    drawDivider(ctx, x1: pA.x, y1: pA.y, x2: pB.x, y2: pB.y, color: withAlpha(cViolet, 0.55), width: 1.4)
                    let uEdge = (t * 3.0 + CGFloat(i + j) * 0.23).truncatingRemainder(dividingBy: 1.0)
                    let ex = pA.x + (pB.x - pA.x) * uEdge
                    let ey = pA.y + (pB.y - pA.y) * uEdge
                    ctx.saveGState()
                    ctx.setFillColor(cCyan)
                    ctx.fillEllipse(in: CGRect(x: ex - 3.0, y: ey - 2.2, width: 6.0, height: 4.4))
                    ctx.restoreGState()
                }
            }
        }
        for (idx, mn) in meshNodes.enumerated() {
            let pt = isoPoint(cx: isoX, cy: layerY[2], u: mn.0, v: mn.1, rx: rx, ry: ry)
            let p = 0.5 + 0.5 * sin(angle * 2.0 - CGFloat(idx) * 0.7)
            ctx.saveGState()
            ctx.setFillColor(withAlpha(cViolet, 0.45 + 0.50 * p))
            ctx.fillEllipse(in: CGRect(x: pt.x - 9, y: pt.y - 6.5, width: 18, height: 13))
            ctx.setFillColor(cInk)
            ctx.fillEllipse(in: CGRect(x: pt.x - 3.5, y: pt.y - 2.5, width: 7, height: 5))
            ctx.restoreGState()
        }

        let f3 = planeFocus(3)
        drawIsoPlane(ctx, cx: isoX, cy: layerY[3], rx: rx, ry: ry, fill: rgba(55, 42, 12, 0.45 + 0.24 * f3), stroke: cAmber, lineWidth: 1.8 + 1.4 * f3)
        for rIdx in 0..<8 {
            let u = -0.7 + CGFloat(rIdx) * 0.2
            let pt = isoPoint(cx: isoX, cy: layerY[3], u: u, v: 0.0, rx: rx, ry: ry)
            let p = 0.5 + 0.5 * sin(angle * 3.0 - CGFloat(rIdx) * 0.65)
            let barH = 6.0 + 18.0 * p
            ctx.saveGState()
            ctx.setFillColor(withAlpha(cAmber, 0.55 + 0.45 * p))
            ctx.fillEllipse(in: CGRect(x: pt.x - 7.5, y: pt.y - 5.5, width: 15, height: 11))
            ctx.setStrokeColor(cAmber)
            ctx.setLineWidth(2.4)
            ctx.move(to: CGPoint(x: pt.x, y: pt.y))
            ctx.addLine(to: CGPoint(x: pt.x, y: pt.y + barH))
            ctx.strokePath()
            ctx.restoreGState()
        }

        let f4 = planeFocus(4)
        drawIsoPlane(ctx, cx: isoX, cy: layerY[4], rx: rx, ry: ry, fill: rgba(16, 52, 32, 0.44 + 0.24 * f4), stroke: cGreen, lineWidth: 2.0 + 1.4 * f4)
        let ringR = 0.22 + 0.48 * ((t * 2.0).truncatingRemainder(dividingBy: 1.0))
        let ringAlpha = 1.0 - ((t * 2.0).truncatingRemainder(dividingBy: 1.0))
        let ringPath = CGMutablePath()
        for step in 0...36 {
            let a = CGFloat(step) / 36.0 * 2.0 * .pi
            let pt = isoPoint(cx: isoX, cy: layerY[4], u: ringR * cos(a), v: ringR * sin(a), rx: rx, ry: ry)
            if step == 0 { ringPath.move(to: pt) } else { ringPath.addLine(to: pt) }
        }
        ctx.saveGState()
        ctx.addPath(ringPath)
        ctx.setStrokeColor(withAlpha(cAmber, 0.75 * ringAlpha))
        ctx.setLineWidth(2.2)
        ctx.strokePath()
        ctx.restoreGState()

        for u in stride(from: -0.65, through: 0.65, by: 0.32) {
            for v in stride(from: -0.65, through: 0.65, by: 0.32) {
                let pt = isoPoint(cx: isoX, cy: layerY[4], u: CGFloat(u), v: CGFloat(v), rx: rx, ry: ry)
                let dist = hypot(CGFloat(u), CGFloat(v))
                let col = dist < 0.45 ? cAmber : (dist < 0.75 ? cGreen : cCyan)
                let rDot: CGFloat = dist < 0.45 ? 7.0 : 5.0
                ctx.saveGState()
                ctx.setFillColor(col)
                ctx.fillEllipse(in: CGRect(x: pt.x - rDot, y: pt.y - rDot * 0.7, width: rDot * 2, height: rDot * 1.4))
                ctx.restoreGState()
            }
        }

        let rightSpecs: [(idx: Int, y: CGFloat, title: String, sub: String, col: CGColor, math: [MathToken])] = [
            (4, 720, "5. Statistical Evaluation & Domain Bias Audit",
             "McNemar + 1,000-bootstrap CI + contract/party/jurisdiction/ALL-CAPS slices",
             cGreen,
             [.op("Macro-F1"), .r(" = "), .frac(num: [.r("1")], den: [.r("|C|")]), .r(" ∑"), .sub("c ∈ C"), .space(4), .i("F"), .sub("1,c"), .space(14), .r("(0.8270 DistilBERT · 0.8291 TF-IDF)", cGreen)]),
            (3, 590, "4. Low-Rank LoRA Adaptation (r = 8, α = 16)",
             "Trainable rank-8 adapters on q_lin & v_lin across all 6 layers (814,948 params)",
             cAmber,
             [.i("h"), .r(" = "), .i("W"), .sub("0"), .i("x"), .r(" + "), .frac(num: [.i("α")], den: [.i("r")]), .i("B A x"), .space(14), .r("(-98.8% params · 0.8144 F1)", cGreen)]),
            (2, 460, "3. Contextual Backbone & Retrieval Index (m4 / m2)",
             "DistilBERT 10-ep encoder + 13 retrieval encodings (TF-IDF, BGE, E5, MiniLM, Legal-BERT)",
             cViolet,
             [.op("cos"), .r(" "), .i("θ"), .r(" = "), .i("q"), .sup("⊤"), .i("d"), .r(" ∈ [-1, 1]   ·   60,000-clause exact matrix search", cViolet)]),
            (1, 330, "2. WordPiece Tokenisation & Length-Grouped Batching",
             "30,522 subwords (0% OOV) · max_length = 512 keeps 98.2% whole",
             cCyan,
             [.i("X"), .r(" ∈ ℤ"), .sup("B × L"), .space(14), .r("(0.565× token cost vs random 256)", cCyan)]),
            (0, 200, "1. LEDGAR Corpus & Official LexGLUE Split",
             "80,000 U.S. SEC clauses across 100 categories (pinned Hub commit c23fdff1)",
             cBlue,
             [.i("N"), .sub("train"), .r(" = 60,000   ·   "), .i("N"), .sub("val"), .r(" = 10,000   ·   "), .i("N"), .sub("test"), .r(" = 10,000")])
        ]

        for rs in rightSpecs {
            let pf = planeFocus(rs.idx)
            drawFlowingLine(ctx, x1: isoX + rx + 14, y1: rs.y, x2: 910, y2: rs.y, color: rs.col, t: t * 2.0 + CGFloat(rs.idx) * 0.2, dots: 1)
            ctx.saveGState()
            ctx.setFillColor(rs.col)
            ctx.fillEllipse(in: CGRect(x: 910 - 5 - 2 * pf, y: rs.y - 5 - 2 * pf, width: 10 + 4 * pf, height: 10 + 4 * pf))
            ctx.restoreGState()

            drawText(ctx, rs.title, x: 935, y: rs.y + 22, size: 25.5, weight: .bold, color: cInk)
            drawText(ctx, rs.sub, x: 935, y: rs.y - 10, size: 19, weight: .medium, color: cSlate)
            drawMath(ctx, rs.math, x: 935, y: rs.y - 46, size: 20, color: rs.col)
        }

    // ------------------------------------------------------------------------
    // SLIDE 04: LEDGAR CORPUS AUDIT & 138× ZIPFIAN CLASS IMBALANCE
    // ------------------------------------------------------------------------
    case 4:
        drawStageCornerGlows(ctx, topLeftColor: cAmber, bottomRightColor: cBlue, t: t, intensity: 0.16)
        drawTwoToneTitle(ctx, leftText: "80,000 clauses & ", rightText: "138× Zipfian imbalance", rightColor: cAmber)
        drawText(ctx, "Auditing lex_glue/ledgar class frequencies (60k train · 10k val · 10k test) explains why accuracy hides rare provisions", x: 960, y: 882, size: 24, weight: .medium, color: cSlate, align: .center)

        let gx0: CGFloat = 130
        let gy0: CGFloat = 230
        let gw: CGFloat  = 840
        let gh: CGFloat  = 480

        drawVectorArrow(ctx, x1: gx0, y1: gy0, x2: gx0 + gw + 25, y2: gy0, color: cSlate, width: 2.2, headLength: 10)
        drawVectorArrow(ctx, x1: gx0, y1: gy0, x2: gx0, y2: gy0 + gh + 25, color: cSlate, width: 2.2, headLength: 10)
        drawMath(ctx, [.op("Training Clauses "), .i("N"), .sub("c")], x: gx0 + 14, y: gy0 + gh + 10, size: 19, color: cInk)
        drawMath(ctx, [.op("Class Rank "), .i("k"), .r(" (1 → 100)")], x: gx0 + gw - 10, y: gy0 - 38, size: 19, color: cInk, align: .right)

        let xR16 = gx0 + (19.0 / 100.0) * gw
        let xR78 = gx0 + (78.0 / 100.0) * gw
        drawDivider(ctx, x1: xR16, y1: gy0, x2: xR16, y2: gy0 + gh - 20, color: withAlpha(cBlue, 0.45), width: 1.6, dashed: true)
        drawDivider(ctx, x1: xR78, y1: gy0, x2: xR78, y2: gy0 + gh - 20, color: withAlpha(cCoral, 0.45), width: 1.6, dashed: true)

        drawText(ctx, "HEAD (16)", x: (gx0 + xR16) / 2, y: gy0 + gh - 28, size: 18, weight: .bold, color: cBlue, align: .center)
        drawText(ctx, "≥ 1,000 train", x: (gx0 + xR16) / 2, y: gy0 + gh - 52, size: 16, weight: .medium, color: cInk, align: .center)

        drawText(ctx, "MIDDLE (62)", x: (xR16 + xR78) / 2, y: gy0 + gh - 28, size: 18, weight: .bold, color: cAmber, align: .center)
        drawText(ctx, "300 to 999 train", x: (xR16 + xR78) / 2, y: gy0 + gh - 52, size: 16, weight: .medium, color: cInk, align: .center)

        drawText(ctx, "TAIL (22)", x: (xR78 + gx0 + gw) / 2, y: gy0 + gh - 28, size: 18, weight: .bold, color: cCoral, align: .center)
        drawText(ctx, "< 300 train", x: (xR78 + gx0 + gw) / 2, y: gy0 + gh - 52, size: 16, weight: .medium, color: cInk, align: .center)

        func zipfCount(_ rank: CGFloat) -> CGFloat {
            let norm = pow(rank / 100.0, 0.42)
            return 3144.0 * pow(1.0 - 0.94 * norm, 2.35) + 23.0
        }
        func zipfY(_ rank: CGFloat) -> CGFloat {
            return gy0 + (zipfCount(rank) / 3350.0) * (gh - 50)
        }

        let scanRank = 16.0 + 70.0 * (0.5 - 0.5 * cos(angle * 2.0 + 0.9))
        let scanInt  = Int(round(scanRank))

        let areaPath = CGMutablePath()
        let activeArea = CGMutablePath()
        let curvePath = CGMutablePath()
        areaPath.move(to: CGPoint(x: gx0, y: gy0))
        activeArea.move(to: CGPoint(x: gx0, y: gy0))
        for k in 1...100 {
            let px = gx0 + (CGFloat(k) / 100.0) * gw
            let py = zipfY(CGFloat(k))
            areaPath.addLine(to: CGPoint(x: px, y: py))
            if k <= scanInt {
                activeArea.addLine(to: CGPoint(x: px, y: py))
            }
            if k == 1 { curvePath.move(to: CGPoint(x: px, y: py)) }
            else { curvePath.addLine(to: CGPoint(x: px, y: py)) }
        }
        areaPath.addLine(to: CGPoint(x: gx0 + gw, y: gy0))
        areaPath.closeSubpath()
        let scanX = gx0 + (CGFloat(scanInt) / 100.0) * gw
        let scanY = zipfY(CGFloat(scanInt))
        activeArea.addLine(to: CGPoint(x: scanX, y: gy0))
        activeArea.closeSubpath()

        let scanCol = scanInt <= 16 ? cBlue : (scanInt <= 78 ? cAmber : cCoral)

        ctx.saveGState()
        ctx.addPath(areaPath)
        ctx.setFillColor(withAlpha(cAmber, 0.08))
        ctx.fillPath()
        ctx.addPath(activeArea)
        ctx.setFillColor(withAlpha(scanCol, 0.20))
        ctx.fillPath()
        ctx.addPath(curvePath)
        ctx.setStrokeColor(cAmber)
        ctx.setLineWidth(3.5)
        ctx.strokePath()
        ctx.restoreGState()

        let yRank1 = zipfY(1)
        let yRank100 = zipfY(100)
        ctx.saveGState()
        ctx.setFillColor(cBlue)
        ctx.fillEllipse(in: CGRect(x: gx0 + gw * 0.01 - 6, y: yRank1 - 6, width: 12, height: 12))
        ctx.setFillColor(cCoral)
        ctx.fillEllipse(in: CGRect(x: gx0 + gw - 6, y: yRank100 - 6, width: 12, height: 12))
        ctx.restoreGState()

        drawText(ctx, "Rank 1: Governing Laws (3,167)", x: gx0 + 20, y: yRank1 - 6, size: 18.5, weight: .bold, color: cBlue)
        drawCard(ctx, x: gx0 + gw - 368, y: yRank100 + 13, w: 366, h: 28, radius: 6, fill: cBg, stroke: nil)
        drawText(ctx, "Rank 100: Books (23 train · 0 val · 2 test)", x: gx0 + gw - 6, y: yRank100 + 20, size: 18, weight: .bold, color: cCoral, align: .right)

        drawDivider(ctx, x1: scanX, y1: gy0, x2: scanX, y2: scanY, color: scanCol, width: 2.0, dashed: true)
        let curCount = Int(round(zipfCount(CGFloat(scanInt))))
        let gradRatio = Double(curCount) / 23.0
        let vecLen: CGFloat = 28.0 + min(1.0, CGFloat(gradRatio / 137.7)) * 95.0
        drawVectorArrow(ctx, x1: scanX, y1: scanY, x2: scanX, y2: scanY + vecLen, color: scanCol, width: 2.6, headLength: 10)

        ctx.saveGState()
        ctx.setFillColor(scanCol)
        ctx.fillEllipse(in: CGRect(x: scanX - 7, y: scanY - 7, width: 14, height: 14))
        ctx.setFillColor(cInk)
        ctx.fillEllipse(in: CGRect(x: scanX - 3, y: scanY - 3, width: 6, height: 6))
        ctx.restoreGState()

        let calloutX = min(max(scanX, gx0 + 210), gx0 + gw - 210)
        let calloutY = scanY + vecLen + 14
        drawCard(ctx, x: calloutX - 205, y: calloutY - 9, w: 410, h: 34, radius: 9, fill: cCardBg, stroke: withAlpha(scanCol, 0.55), lineWidth: 1.3)
        drawMath(ctx, [
            .r("Rank "), .i("k"), .r(" = \(scanInt)  ·  "),
            .i("N"), .sub("k"), .r(" = \(curCount)  ·  "),
            .r(String(format: "%.1f×", gradRatio), scanCol), .r(" vs Books (23)")
        ], x: calloutX, y: calloutY, size: 18, color: cInk, align: .center)

        drawMath(ctx, [
            .r("Unweighted Gradient:   ∇"), .sub("θ"), .i("L"), .r(" = ∑"),
            .subSup(sub: "c=1", sup: "100"), .space(4),
            .frac(num: [.i("N"), .sub("c")], den: [.i("N")]),
            .r(" ∇"), .sub("θ"), .i("ℓ"), .sub("c"),
            .space(18), .r("→  Governing Laws (3,167) has 137.7× the gradient pull of Books (23)!", cAmber)
        ], x: 115, y: 140, size: 19.5, color: cInk)

        drawKeynoteCard(ctx, x: 1030, y: 495, w: 780, h: 305, radius: 30, accent: cBlue, headerHeight: 88, focus: columnFocus(index: 0, count: 2, t: t))
        drawText(ctx, "Support-Aware Macro-F1", x: 1420, y: 754, size: 26, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "Primary Decision Metric (ledgar.metrics.macro_f1)", x: 1420, y: 722, size: 18.5, weight: .medium, color: cBlue, align: .center)

        drawMath(ctx, [
            .op("Macro-F1"), .r(" = "),
            .frac(num: [.r("1")], den: [.r("|"), .i("C"), .sub("present"), .r("|")]),
            .r(" ∑"), .sub("c ∈ C"), .space(4),
            .frac(
                num: [.r("2 · "), .op("Prec"), .sub("c"), .r(" · "), .op("Rec"), .sub("c")],
                den: [.op("Prec"), .sub("c"), .r(" + "), .op("Rec"), .sub("c")]
            )
        ], x: 1420, y: 632, size: 22.0, color: cCyan, align: .center)

        drawFeatureRow(ctx, icon: .sparkle, iconColor: cBlue, spans: [
            ("Averages over ", .medium, cSlate), ("99 present types on Val", .bold, cBlue), (" (Books = 0) & 100 on Test", .medium, cSlate)
        ], x: 1066, y: 562, size: 18.0)
        drawFeatureRow(ctx, icon: .shield, iconColor: cInk, spans: [
            ("Lazy majority floor: ", .medium, cSlate), ("4.94% Val / 5.82% Test Acc → 0.001 Macro-F1", .bold, cInk)
        ], x: 1066, y: 520, size: 18.0)

        drawKeynoteCard(ctx, x: 1030, y: 165, w: 780, h: 305, radius: 30, accent: cGreen, headerHeight: 88, focus: columnFocus(index: 1, count: 2, t: t))
        drawText(ctx, "60,000 Train · 10,000 Val · 10,000 Test", x: 1420, y: 424, size: 26, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "Official LexGLUE Split Preserved · Hub Commit c23fdff1", x: 1420, y: 392, size: 18.5, weight: .medium, color: cGreen, align: .center)

        drawText(ctx, "71 of 100 types < 1% of train   ·   282 test clauses (2.8%) copied from train", x: 1420, y: 312, size: 19.0, weight: .bold, color: cGreen, align: .center)

        drawFeatureRow(ctx, icon: .sliders, iconColor: cGreen, spans: [
            ("Test read ", .medium, cSlate), ("strictly once per final model", .bold, cGreen), (" (enforced by test_hygiene.py)", .medium, cSlate)
        ], x: 1066, y: 242, size: 18.0)
        drawFeatureRow(ctx, icon: .check, iconColor: cInk, spans: [
            ("6 duplicate texts carry ", .medium, cSlate), ("2 near-synonym labels", .bold, cInk), (" (Waivers / Governing Law)", .medium, cSlate)
        ], x: 1066, y: 198, size: 18.0)

    // ------------------------------------------------------------------------
    // SLIDE 05: TOKENISATION & LENGTH-GROUPED BATCH PACKING
    // ------------------------------------------------------------------------
    case 5:
        drawStageCornerGlows(ctx, topLeftColor: cCyan, bottomRightColor: cBlue, t: t, intensity: 0.16)
        drawTwoToneTitle(ctx, leftText: "Subword precision ", rightText: "& length-grouped batching", rightColor: cCyan)
        drawText(ctx, "Keeping 98.2% of clauses whole at max_length = 512 while cutting token compute by 43.5% vs random 256", x: 960, y: 882, size: 24, weight: .medium, color: cSlate, align: .center)

        drawMath(ctx, [
            .i("L"), .sub("batch"), .r(" = "), .op("min"), .r("(512,  "), .op("max"), .sub("i ∈ B"), .r(" |"), .i("x"), .sub("i"), .r("|)   →   "),
            .r("8.67M tokens/ep (0.565× vs random 256 · -68.1% vs random 512)", cCyan)
        ], x: 120, y: 748, size: 19.5, color: cInk)

        let seqLens: [CGFloat] = [0.58, 0.52, 0.46, 0.39, 0.33, 0.25, 0.19]
        let maxBatchFrac: CGFloat = 0.58
        let barX: CGFloat = 225
        let barW: CGFloat = 740

        let attnScanFrac = maxBatchFrac * (0.5 - 0.5 * cos(angle * 2.0 + 0.6))
        let attnScanX = barX + barW * attnScanFrac

        for (r, frac) in seqLens.enumerated() {
            let ry: CGFloat = 650 - CGFloat(r) * 52
            drawMath(ctx, [.i("x"), .sub("\(r+1)")], x: 170, y: ry + 7, size: 20, color: cInk)

            let actW = barW * frac
            let isScanned = attnScanFrac <= frac
            drawCard(ctx, x: barX, y: ry, w: actW, h: 32, radius: 8, fill: withAlpha(cBlue, isScanned ? 0.88 : 0.72), stroke: isScanned ? withAlpha(cCyan, 0.75) : nil, lineWidth: 1.2)

            let dynPadW = barW * (maxBatchFrac - frac)
            if dynPadW > 4 {
                drawCard(ctx, x: barX + actW + 4, y: ry, w: dynPadW - 4, h: 32, radius: 8, fill: withAlpha(cCyan, 0.24), stroke: nil)
            }

            let savedX = barX + barW * maxBatchFrac + 8
            let savedW = barW * (1.0 - maxBatchFrac) - 8
            let pulseElim = 0.16 + 0.08 * (0.5 + 0.5 * sin(angle * 2.0))
            drawCard(ctx, x: savedX, y: ry, w: savedW, h: 32, radius: 8, fill: rgba(255, 107, 107, pulseElim), stroke: withAlpha(cCoral, 0.38), lineWidth: 1.0)
        }

        drawDivider(ctx, x1: attnScanX, y1: 330, x2: attnScanX, y2: 690, color: withAlpha(cInk, 0.75), width: 2.0)
        ctx.saveGState()
        ctx.setFillColor(cInk)
        ctx.fillEllipse(in: CGRect(x: attnScanX - 5, y: 690 - 5, width: 10, height: 10))
        ctx.restoreGState()

        let cutX = barX + barW * maxBatchFrac
        drawDivider(ctx, x1: cutX, y1: 315, x2: cutX, y2: 690, color: cCyan, width: 2.4, dashed: true)
        drawMath(ctx, [.i("L"), .sub("batch"), .r(" = 296")], x: cutX, y: 280, size: 19.5, color: cCyan, align: .center)
        drawText(ctx, "512 Max", x: barX + barW, y: 280, size: 19, weight: .bold, color: cCoral, align: .right)
        drawText(ctx, "Eliminated [PAD] Columns (-68.1% vs Random 512)", x: barX + barW, y: 702, size: 17.5, weight: .bold, color: cCoral, align: .right)

        drawText(ctx, "WordPiece Length Audit: Median = 106 · P90 = 291 · P95 = 381 · P99 = 589 · 98.2% whole at 512.", x: 120, y: 195, size: 19, weight: .medium, color: cInk)

        drawKeynoteCard(ctx, x: 1030, y: 495, w: 780, h: 305, radius: 30, accent: cBlue, headerHeight: 88, focus: columnFocus(index: 0, count: 2, t: t))
        drawText(ctx, "WordPiece Tokeniser & OrderedEvalTrainer", x: 1420, y: 754, size: 26, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "30,522 Subwords · 0% OOV · Sequential Eval Sampler Fix", x: 1420, y: 722, size: 18.5, weight: .medium, color: cBlue, align: .center)

        let subStep = Int(floor((t * 6.0).truncatingRemainder(dividingBy: 3.0)))
        drawCard(ctx, x: 1066, y: 614, w: 220, h: 44, radius: 10, fill: rgba(255, 255, 255, 0.13), stroke: cStroke, lineWidth: 1.2)
        drawText(ctx, "\"indemnification\"", x: 1176, y: 629, size: 18.5, weight: .mono, color: cInk, align: .center)
        drawFlowingLine(ctx, x1: 1294, y1: 636, x2: 1348, y2: 636, color: cCyan, t: t * 3.0, dots: 1)

        drawCard(ctx, x: 1356, y: 614, w: 126, h: 44, radius: 10, fill: cBlueTint, stroke: subStep == 0 ? cBlue : withAlpha(cBlue, 0.45), lineWidth: 1.8)
        drawText(ctx, "indemn", x: 1419, y: 629, size: 18.5, weight: .mono, color: cBlue, align: .center)
        drawCard(ctx, x: 1494, y: 614, w: 120, h: 44, radius: 10, fill: cCyanTint, stroke: subStep == 1 ? cCyan : withAlpha(cCyan, 0.45), lineWidth: 1.8)
        drawText(ctx, "##ific", x: 1554, y: 629, size: 18.5, weight: .mono, color: cCyan, align: .center)
        drawCard(ctx, x: 1626, y: 614, w: 146, h: 44, radius: 10, fill: cGreenTint, stroke: subStep == 2 ? cGreen : withAlpha(cGreen, 0.45), lineWidth: 1.8)
        drawText(ctx, "##ation", x: 1699, y: 629, size: 18.5, weight: .mono, color: cGreen, align: .center)

        drawFeatureRow(ctx, icon: .sparkle, iconColor: cBlue, spans: [
            ("At 256 tokens, ", .medium, cSlate), ("69% of Capitalization clauses", .bold, cCoral), (" lose their operative tail", .medium, cSlate)
        ], x: 1066, y: 560, size: 18.0)
        drawFeatureRow(ctx, icon: .check, iconColor: cGreen, spans: [
            ("OrderedEvalTrainer ", .bold, cGreen), ("prevents group_by_length from scrambling test preds", .medium, cSlate)
        ], x: 1066, y: 518, size: 18.0)

        drawKeynoteCard(ctx, x: 1030, y: 165, w: 780, h: 305, radius: 30, accent: cAmber, headerHeight: 88, focus: columnFocus(index: 1, count: 2, t: t))
        drawText(ctx, "Regex Word Tokeniser + GloVe 50d (m3)", x: 1420, y: 424, size: 26, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "13,252 Vocabulary (min_freq = 2, 18,745 raw) · max_words = 476 (P99)", x: 1420, y: 392, size: 18.5, weight: .medium, color: cAmber, align: .center)

        drawText(ctx, "88.9% GloVe coverage   ·   11.1% rare legal words random   ·   0.26% Val <UNK>", x: 1420, y: 312, size: 19.0, weight: .bold, color: cInk, align: .center)

        drawFeatureRow(ctx, icon: .bolt, iconColor: cGreen, spans: [
            ("Length-grouped mega-batches cut MPS epoch ", .medium, cSlate), ("8.6 min → 3.2 min (2.7×)", .bold, cGreen)
        ], x: 1066, y: 242, size: 18.0)
        drawFeatureRow(ctx, icon: .sliders, iconColor: cAmber, spans: [
            ("Static 50d embeddings plateau at ", .medium, cSlate), ("0.7895 Val / 0.7802 Test Macro-F1", .bold, cInk)
        ], x: 1066, y: 198, size: 18.0)

    // ------------------------------------------------------------------------
    // SLIDE 06: FOUR MODEL FAMILIES UNDER ONE VALIDATION PROTOCOL
    // ------------------------------------------------------------------------
    case 6:
        drawStageCornerGlows(ctx, topLeftColor: cBlue, bottomRightColor: cGreen, t: t, intensity: 0.16)
        drawTwoToneTitle(ctx, leftText: "Four model families under ", rightText: "one validation protocol", rightColor: cBlue)
        drawText(ctx, "Held-out test results (10,000 clauses, read once per model) across our four milestone architectures", x: 960, y: 882, size: 24, weight: .medium, color: cSlate, align: .center)

        let cards: [(name: String, sub: String, lead1: String, lead2: String, math: [MathToken], r1: [(String, FontWeight, CGColor)], r2: [(String, FontWeight, CGColor)], r3: [(String, FontWeight, CGColor)], foot: String, col: CGColor)] = [
            ("Lazy Majority", "0.0011 Test Macro-F1",
             "Zero-skill floor predicting", "the top class on all clauses",
             [.r("∀"), .i("x"), .r(" : "), .i("y"), .r(" = "), .op("argmax"), .sub("c"), .r(" "), .i("N"), .sub("c")],
             [("Always predicts ", .medium, cSlate), ("Governing Laws", .bold, cInk)],
             [("Val Macro-F1 ", .medium, cSlate), ("0.0010 (4.94% Acc)", .bold, cInk)],
             [("Test Accuracy ", .medium, cSlate), ("5.82%", .bold, cInk)],
             "*0 trainable parameters", cSlate),

            ("TF-IDF + LogReg", "0.8291 Test Macro-F1",
             "Sublinear (1,2)-gram features", "+ L2 multinomial logistic head",
             [.i("P"), .r("("), .i("y"), .r("|"), .i("x"), .r(") = "), .op("softmax"), .r("("), .i("W φ"), .r("("), .i("x"), .r(") + "), .i("b"), .r(")")],
             [("173,477 ", .bold, cBlue), ("n-grams (min_df=2, C=30)", .medium, cSlate)],
             [("Val Macro-F1 ", .medium, cSlate), ("0.8265", .bold, cBlue)],
             [("Test Accuracy ", .medium, cSlate), ("87.71%", .bold, cInk)],
             "*17.35M weights · 2.3 min on CPU", cBlue),

            ("BiLSTM + GloVe", "0.7802 ± 0.0055 Test F1",
             "1-layer bidirectional LSTM", "packed final hidden state",
             [.i("h"), .r(" = ["), .i("h"), .subSup(sub: "L(x)", sup: "→"), .r(" ∥ "), .i("h"), .subSup(sub: "1", sup: "←"), .r("] ∈ ℝ"), .sup("512")],
             [("Val Macro-F1 ", .medium, cSlate), ("0.7895 ± 0.0044", .bold, cAmber)],
             [("Hidden H=256 ", .medium, cSlate), ("+ √inv weights", .bold, cGreen)],
             [("Test Accuracy ", .medium, cSlate), ("84.86% (3 seeds)", .bold, cInk)],
             "*1.34M params · 32.6 min on MPS", cAmber),

            ("DistilBERT & LoRA", "0.8270 ± 0.0036 Test F1",
             "10-epoch Full Fine-Tuning", "vs Rank-8 LoRA adapters",
             [.i("h"), .r(" = "), .i("W"), .sub("0"), .i("x"), .r(" + "), .frac(num: [.i("α")], den: [.i("r")]), .i("B A x")],
             [("Full FT: ", .medium, cSlate), ("0.8270 F1 · 87.98% Acc", .bold, cGreen)],
             [("LoRA r=8: ", .medium, cSlate), ("0.8144 F1 · 87.49% Acc", .bold, cCyan)],
             [("LoRA footprint ", .medium, cSlate), ("0.81M params (1.2%)", .bold, cInk)],
             "*11.6 min on A100 · ~3 MB LoRA adapter", cGreen)
        ]

        let cardW: CGFloat = 415
        let gap: CGFloat   = 24
        let startX: CGFloat = 94

        for (i, c) in cards.enumerated() {
            let f = columnFocus(index: i, count: 4, t: t)
            let x = startX + CGFloat(i) * (cardW + gap)
            let cx = x + cardW / 2.0

            drawKeynoteCard(ctx, x: x, y: 185, w: cardW, h: 630, radius: 34, accent: c.col, headerHeight: 132, focus: f)
            drawText(ctx, c.name, x: cx, y: 750, size: 28, weight: .bold, color: cInk, align: .center)
            drawText(ctx, c.sub, x: cx, y: 710, size: 20.5, weight: .bold, color: i == 0 ? cInk : c.col, align: .center)

            drawText(ctx, c.lead1, x: cx, y: 642, size: 18.0, weight: .medium, color: cInk, align: .center)
            drawText(ctx, c.lead2, x: cx, y: 615, size: 18.0, weight: .medium, color: cSlate, align: .center)
            drawCard(ctx, x: x + 22, y: 498, w: cardW - 44, h: 68, radius: 14, fill: rgba(255, 255, 255, 0.045), stroke: nil)
            drawMath(ctx, c.math, x: cx, y: 526, size: 21.5, color: cInk, align: .center)

            drawFeatureRow(ctx, icon: .sparkle, iconColor: c.col, spans: c.r1, x: x + 24, y: 438, size: 18.0)
            drawFeatureRow(ctx, icon: .sliders, iconColor: cInk, spans: c.r2, x: x + 24, y: 380, size: 18.0)
            drawFeatureRow(ctx, icon: .bolt, iconColor: c.col, spans: c.r3, x: x + 24, y: 322, size: 18.0)

            drawText(ctx, c.foot, x: cx, y: 215, size: 14.5, weight: .regular, color: cMuted, align: .center)
        }

        drawMath(ctx, [
            .r("Parameter Footprint:   "),
            .r("LoRA ", cGreen), .i("r", cGreen), .r("=8 (0.81M)", cGreen),
            .r("   <   "),
            .r("BiLSTM ", cAmber), .i("H", cAmber), .r("=256 (1.34M)", cAmber),
            .r("   ≪   "),
            .r("TF-IDF Linear (17.35M)", cBlue),
            .r("   ≪   "),
            .r("DistilBERT Full FT (67.03M)", cInk)
        ], x: 960, y: 112, size: 20.5, color: cInk, align: .center)

    // ------------------------------------------------------------------------
    // SLIDE 07: CLASSICAL LINEAR BASELINE — TF-IDF + LOGISTIC REGRESSION
    // ------------------------------------------------------------------------
    case 7:
        drawStageCornerGlows(ctx, topLeftColor: cArgonElectric, bottomRightColor: cCyan, t: t, intensity: 0.17)
        drawTwoToneTitle(ctx, leftText: "TF-IDF + LogReg sets a ", rightText: "0.8291 test bar", rightColor: cArgonElectric)
        drawText(ctx, "Sublinear (1,2)-gram features + L2-regularised multinomial logistic regression reach 0.8265 Val / 0.8291 Test Macro-F1", x: 960, y: 882, size: 24, weight: .medium, color: cSlate, align: .center)

        drawMath(ctx, [
            .op("tfidf"), .r("("), .i("t, d"), .r(") = (1 + "), .op("log"), .r(" "), .op("tf"), .r("("), .i("t, d"), .r(")) · "),
            .op("log"), .frac(num: [.i("N")], den: [.op("df"), .r("("), .i("t"), .r(")")]),
            .r("   ∈ ℝ"), .sup("173,477")
        ], x: 150, y: 762, size: 23.5, color: cCyan)

        let ox: CGFloat = 190
        let oy: CGFloat = 250
        let aw: CGFloat = 660
        let ah: CGFloat = 430
        drawVectorArrow(ctx, x1: ox, y1: oy, x2: ox + aw, y2: oy, color: cSlate, width: 2.2, headLength: 10)
        drawVectorArrow(ctx, x1: ox, y1: oy, x2: ox, y2: oy + ah, color: cSlate, width: 2.2, headLength: 10)
        drawMath(ctx, [.i("φ"), .sub("1"), .r("("), .i("x"), .r(") : "), .op("tfidf"), .r("(\"governed by\")")], x: ox + aw - 10, y: oy - 36, size: 19.5, color: cInk, align: .right)
        drawMath(ctx, [.i("φ"), .sub("2"), .r("("), .i("x"), .r(") : "), .op("tfidf"), .r("(\"written notice\")")], x: ox + 18, y: oy + ah + 2, size: 19.5, color: cInk)

        let hx1: CGFloat = ox + 90
        let hy1: CGFloat = oy + 40
        let hx2: CGFloat = ox + 620
        let hy2: CGFloat = oy + 370
        let hdx = hx2 - hx1
        let hdy = hy2 - hy1
        let hLen2 = hdx * hdx + hdy * hdy

        let marginShift: CGFloat = 1.0 + 0.16 * sin(angle)
        drawDivider(ctx, x1: hx1 - 25 * marginShift, y1: hy1 + 15 * marginShift, x2: hx2 - 25 * marginShift, y2: hy2 + 15 * marginShift, color: withAlpha(cCyan, 0.35), width: 1.4, dashed: true)
        drawDivider(ctx, x1: hx1 + 25 * marginShift, y1: hy1 - 15 * marginShift, x2: hx2 + 25 * marginShift, y2: hy2 - 15 * marginShift, color: withAlpha(cCyan, 0.35), width: 1.4, dashed: true)
        drawDivider(ctx, x1: hx1, y1: hy1, x2: hx2, y2: hy2, color: cCyan, width: 3.0)

        let midHX: CGFloat = ox + 355
        let midHY: CGFloat = oy + 205
        drawVectorArrow(ctx, x1: midHX, y1: midHY, x2: midHX - 62, y2: midHY + 100, color: cAmber, width: 3.0, headLength: 12)
        drawMath(ctx, [.b("w", cAmber), .sub("c", cAmber), .sup("⊤", cAmber), .i("φ"), .r("("), .i("x"), .r(") + "), .i("b"), .sub("c"), .r(" = 0")], x: midHX + 46, y: midHY - 14, size: 22, color: cCyan)

        let termPts: [(CGFloat, CGFloat)] = [(250, 480), (295, 535), (340, 500), (270, 575), (390, 560), (435, 610), (355, 620)]
        let govPts: [(CGFloat, CGFloat)] = [(480, 310), (540, 345), (595, 315), (650, 385), (558, 388), (690, 340), (630, 290)]

        let activePtIdx = Int(floor(t * 14.0)) % 14

        for (idx, pt) in termPts.enumerated() {
            let u = ((pt.0 - hx1) * hdx + (pt.1 - hy1) * hdy) / hLen2
            let px = hx1 + u * hdx
            let py = hy1 + u * hdy
            let isSelected = (idx == activePtIdx)
            drawDivider(ctx, x1: pt.0, y1: pt.1, x2: px, y2: py, color: withAlpha(cGreen, isSelected ? 0.85 : 0.28), width: isSelected ? 2.4 : 1.1, dashed: !isSelected)
            if isSelected {
                drawRadialGlow(ctx, cx: pt.0, cy: pt.1, radius: 28, color: cGreen, alpha: 0.45)
                ctx.saveGState()
                ctx.setStrokeColor(cInk)
                ctx.setLineWidth(2.0)
                ctx.strokeEllipse(in: CGRect(x: pt.0 - 10, y: pt.1 - 10, width: 20, height: 20))
                ctx.restoreGState()
            }
            ctx.saveGState()
            ctx.setFillColor(cGreen)
            ctx.fillEllipse(in: CGRect(x: pt.0 - 6.5, y: pt.1 - 6.5, width: 13, height: 13))
            ctx.restoreGState()
        }
        drawText(ctx, "Terminations (0.91 F1)", x: 245, y: 640, size: 19, weight: .bold, color: cGreen)

        for (idx, pt) in govPts.enumerated() {
            let u = ((pt.0 - hx1) * hdx + (pt.1 - hy1) * hdy) / hLen2
            let px = hx1 + u * hdx
            let py = hy1 + u * hdy
            let isSelected = ((idx + 7) == activePtIdx)
            drawDivider(ctx, x1: pt.0, y1: pt.1, x2: px, y2: py, color: withAlpha(cArgonIce, isSelected ? 0.85 : 0.28), width: isSelected ? 2.4 : 1.1, dashed: !isSelected)
            if isSelected {
                drawRadialGlow(ctx, cx: pt.0, cy: pt.1, radius: 28, color: cArgonElectric, alpha: 0.45)
                ctx.saveGState()
                ctx.setStrokeColor(cInk)
                ctx.setLineWidth(2.0)
                ctx.strokeEllipse(in: CGRect(x: pt.0 - 10, y: pt.1 - 10, width: 20, height: 20))
                ctx.restoreGState()
            }
            ctx.saveGState()
            ctx.setFillColor(cArgonElectric)
            ctx.fillEllipse(in: CGRect(x: pt.0 - 6.5, y: pt.1 - 6.5, width: 13, height: 13))
            ctx.restoreGState()
        }
        drawText(ctx, "Governing Laws (0.96 F1)", x: 495, y: 262, size: 19, weight: .bold, color: cArgonIce)

        drawText(ctx, "Test by Band: Head (≥1k) = 0.958 F1 · Middle = 0.870 · Tail (<300) = 0.749", x: 145, y: 155, size: 18.5, weight: .bold, color: cInk)

        let findings: [(metric: String, title: String, desc1: String, desc2: String, math: [MathToken], col: CGColor)] = [
            ("+3.4 pts Macro-F1", "Bigrams Capture Legal Phraseology",
             "Expanding from unigrams (0.7922 Val F1) to (1,2)-grams (173,477 features)",
             "captures multi-word legal idioms like 'governed by' and 'prior written notice'.",
             [.op("n-gram range"), .r(" = (1, 2)   ·   "), .op("min df"), .r(" = 2   ·   "), .op("sublinear tf"), .r(" = True")],
             cAmber),
            ("0.8265 vs 0.8262", "Why Unweighted Beats Balanced L-BFGS",
             "Inverse-frequency weights up to 138× amplify noisy rare-class n-grams",
             "and prevent saga convergence; unweighted L-BFGS wins on validation.",
             [.op("Val Macro-F1"), .r(" :  0.8265 (unweighted)  >  0.8262 (balanced)")],
             cArgonElectric),
            ("Zero Test Leakage", "Regularisation Hygiene at C = 30",
             "Swept C ∈ {0.3, 1, 3, 10, 30} on Val (0.774 @ C=1 → 0.819 @ C=10 → 0.827 @ C=30);",
             "grid frozen once test predictions were written to prevent optimistic leakage.",
             [.op("Test Macro-F1"), .r(" = 0.8291   ·   "), .op("Test Accuracy"), .r(" = 87.71% (139s CPU)")],
             cGreen)
        ]

        for (i, fd) in findings.enumerated() {
            let f = columnFocus(index: i, count: 3, t: t)
            let cy: CGFloat = 692 - CGFloat(i) * 222
            drawKeynoteCard(ctx, x: 960, y: cy - 92, w: 840, h: 202, radius: 26, accent: fd.col, headerHeight: 62, focus: f)
            drawText(ctx, fd.title, x: 992, y: cy + 68, size: 23.5, weight: .bold, color: cInk)
            drawText(ctx, fd.metric, x: 1768, y: cy + 68, size: 20, weight: .bold, color: fd.col, align: .right)
            drawText(ctx, fd.desc1, x: 992, y: cy + 15, size: 18.5, weight: .medium, color: cInk)
            drawText(ctx, fd.desc2, x: 992, y: cy - 13, size: 18.5, weight: .regular, color: cSlate)
            drawMath(ctx, fd.math, x: 992, y: cy - 60, size: 19.5, color: fd.col)
        }

    // ------------------------------------------------------------------------
    // SLIDE 08: NEURAL SEQUENCE BASELINE — BILSTM + GLOVE 50D
    // ------------------------------------------------------------------------
    case 8:
        drawStageCornerGlows(ctx, topLeftColor: cAmber, bottomRightColor: cCyan, t: t, intensity: 0.16)
        drawTwoToneTitle(ctx, leftText: "BiLSTM + GloVe 50d: ", rightText: "packed final hidden states", rightColor: cAmber)
        drawText(ctx, "One-factor-at-a-time validation search (8 ep) followed by 15-epoch multi-seed evaluation (0.7802 Test F1)", x: 960, y: 882, size: 24, weight: .medium, color: cSlate, align: .center)

        drawText(ctx, "4. Linear(512 → 100) + Mean-Normalised √Inverse Class Weights", x: 140, y: 758, size: 22, weight: .bold, color: cViolet)
        drawMath(ctx, [
            .i("w"), .sub("c"), .r(" = "), .op("clip"), .r("("),
            .frac(
                num: [.i("N"), .subSup(sub: "c", sup: "-1/2")],
                den: [.frac(num: [.r("1")], den: [.i("K")]), .r(" ∑"), .sub("j"), .space(3), .i("N"), .subSup(sub: "j", sup: "-1/2")]
            ),
            .r(",  0.5,  5.0)     →   0.7895 Val / 0.7802 Test F1", cGreen)
        ], x: 140, y: 695, size: 20.5, color: cInk)

        drawFlowingLine(ctx, x1: 480, y1: 625, x2: 480, y2: 670, color: cCyan, t: t, dots: 1)

        drawText(ctx, "3. Concatenated Final Hidden States (pack_padded_sequence)", x: 140, y: 588, size: 22, weight: .bold, color: cCyan)
        drawMath(ctx, [
            .i("h"), .r(" = ["),
            .i("h"), .subSup(sub: "L(x)", sup: "→"),
            .r(" ∥ "),
            .i("h"), .subSup(sub: "1", sup: "←"),
            .r("] ∈ ℝ"), .sup("512"),
            .space(18),
            .r("where  pack_padded_sequence  skips  [PAD]  steps")
        ], x: 140, y: 515, size: 21.0, color: cInk)

        let poolTargetX: CGFloat = 368
        let poolTargetY: CGFloat = 480
        drawFlowingLine(ctx, x1: 180, y1: 370, x2: poolTargetX, y2: poolTargetY, color: cCyan, t: t, dots: 1)
        drawFlowingLine(ctx, x1: 180 + 3 * 124, y1: 370, x2: poolTargetX, y2: poolTargetY, color: cAmber, t: t + 0.35, dots: 1)
        for idx in 4..<6 {
            let tx: CGFloat = 180 + CGFloat(idx) * 124
            drawDivider(ctx, x1: tx, y1: 370, x2: tx, y2: 404, color: withAlpha(cCoral, 0.55), width: 1.7, dashed: true)
            drawKeynoteIcon(ctx, .cross, cx: tx, cy: 412, size: 7.5, color: cCoral)
        }

        let tokLabels = ["\"shall\"", "\"govern\"", "\"this\"", "\"deal\"", "[PAD]", "[PAD]"]
        let masks     = ["h←₁", "t=2", "t=3", "h→L(x)", "skip", "skip"]
        let fwdStep = (t * 4.0).truncatingRemainder(dividingBy: 4.0)
        let bwdStep = 3.0 - fwdStep

        for idx in 0..<6 {
            let tx: CGFloat = 180 + CGFloat(idx) * 124
            let isPad = idx >= 4
            let isFwdActive = !isPad && abs(CGFloat(idx) - fwdStep) < 0.75
            let isBwdActive = !isPad && abs(CGFloat(idx) - bwdStep) < 0.75
            let col = isPad ? cCoral : (isFwdActive ? cAmber : (isBwdActive ? cCyan : cAmber))
            let tint = isPad ? rgba(255, 107, 107, 0.14) : ((isFwdActive || isBwdActive) ? rgba(251, 188, 4, 0.28) : cAmberTint)
            let strk: CGColor? = isPad ? withAlpha(cCoral, 0.45) : ((isFwdActive || isBwdActive) ? (isFwdActive ? cAmber : cCyan) : nil)
            drawCard(ctx, x: tx - 51, y: 329, w: 102, h: 38, radius: 10, fill: tint, stroke: strk, lineWidth: 1.6)
            drawText(ctx, isPad ? "unpacked" : "h\(idx+1)", x: tx, y: 342, size: 17, weight: .mono, color: col, align: .center)
            drawText(ctx, tokLabels[idx], x: tx, y: 298, size: 17, weight: .mono, color: isPad ? cSlate : cInk, align: .center)
            drawText(ctx, masks[idx], x: tx, y: 270, size: 16.5, weight: .bold, color: isPad ? cCoral : ((idx == 0 || idx == 3) ? cGreen : cSlate), align: .center)
            if idx < 3 {
                drawVectorArrow(ctx, x1: tx + 53, y1: 354, x2: tx + 71, y2: 354, color: cAmber, width: 1.8, headLength: 7)
                drawVectorArrow(ctx, x1: tx + 71, y1: 342, x2: tx + 53, y2: 342, color: cCyan, width: 1.8, headLength: 7)
            }
        }

        drawText(ctx, "2. Packed 1-Layer BiLSTM (H = 256 per direction → 512d)", x: 140, y: 224, size: 22, weight: .bold, color: cAmber)
        drawText(ctx, "1. Pretrained GloVe 6B 50d Embedding Lookup", x: 140, y: 180, size: 21.5, weight: .bold, color: cArgonIce)
        drawText(ctx, "13,252 vocab (min_freq = 2, 18,745 raw) · 88.9% coverage · 1.34M params", x: 140, y: 150, size: 17.5, weight: .medium, color: cInk)

        let runs: [(label: String, sub: String, f1: CGFloat, score: String, tag: String, col: CGColor)] = [
            ("Final 15-Epoch · Seeds {0, 1, 2}", "Test F1 = 0.7802 ± 0.0055 · Test Acc = 84.86% (best ep 15, 15, 12)", 0.7895, "0.7895", "Final 3-Seed Val", cGreen),
            ("Search Step 4 · + √inv Class Weights", "8-epoch search winner (lr = 2e-3 · dropout = 0.3)", 0.7764, "0.7764", "Search Winner", cArgonElectric),
            ("Search Step 3 · + Hidden State H = 256", "Doubling hidden width from 128 to 256 per direction", 0.7753, "0.7753", "H = 256", cCyan),
            ("Search Step 2 · + Bidirectional Reading", "+2.70 pts Val F1 — only factor worth > 0.5 pt on its own", 0.7724, "0.7724", "+2.70 pts Jump", cAmber),
            ("Search Step 1 · Unidirectional + GloVe 50d", "Matches 100d random init (0.7452) with half the parameters", 0.7454, "0.7454", "1-Dir Baseline", cCoral)
        ]

        for (i, r) in runs.enumerated() {
            let ry: CGFloat = 720 - CGFloat(i) * 122
            let isWin = i == 0
            drawKeynoteCard(ctx, x: 955, y: ry - 48, w: 845, h: 108, radius: 22, accent: r.col, headerHeight: nil, focus: isWin ? 0.95 : 0.25)
            drawText(ctx, r.label, x: 988, y: ry + 24, size: 20.5, weight: .bold, color: cInk)
            drawText(ctx, r.sub, x: 988, y: ry - 3, size: 17, weight: .medium, color: cSlate)

            let barMaxW: CGFloat = 490
            let barW = barMaxW * (r.f1 / 0.85)
            drawCard(ctx, x: 988, y: ry - 30, w: barMaxW, h: 10, radius: 5, fill: rgba(255, 255, 255, 0.07), stroke: nil)
            drawCard(ctx, x: 988, y: ry - 30, w: barW, h: 10, radius: 5, fill: r.col, stroke: nil)

            drawText(ctx, r.score, x: 1765, y: ry + 12, size: 30, weight: .bold, color: r.col, align: .right)
            drawText(ctx, r.tag, x: 1765, y: ry - 20, size: 16, weight: .bold, color: r.col, align: .right)
        }

    // ------------------------------------------------------------------------
    // SLIDE 09: LOW-RANK ADAPTATION (LORALINEAR) MATHEMATICAL GEOMETRY
    // ------------------------------------------------------------------------
    case 9:
        drawStageCornerGlows(ctx, topLeftColor: cViolet, bottomRightColor: cGreen, t: t, intensity: 0.18)
        drawTwoToneTitle(ctx, leftText: "Low-rank adaptation: ", rightText: "98.8% fewer trained parameters", rightColor: cViolet)
        drawText(ctx, "Hand-built LoRALinear (src/ledgar/models/lora.py) verified against HuggingFace peft to the exact parameter count", x: 960, y: 882, size: 24, weight: .medium, color: cSlate, align: .center)

        drawMath(ctx, [
            .i("h"), .r(" = "), .i("W"), .sub("0"), .i("x"), .r(" + "),
            .frac(num: [.i("α")], den: [.i("r")]),
            .i("B A x"), .r("   =   ("), .i("W"), .sub("0"), .r(" + 2.0 · "), .i("B A"), .r(") "), .i("x"), .r("   ∈ ℝ"), .sup("768")
        ], x: 485, y: 765, size: 24.0, color: cInk, align: .center)

        drawCard(ctx, x: 295, y: 200, w: 390, h: 50, radius: 14, fill: cBlueTint, stroke: withAlpha(cArgonElectric, 0.55), lineWidth: 1.4)
        drawMath(ctx, [.r("Input Activation   "), .i("x"), .r(" ∈ ℝ"), .sup("768")], x: 490, y: 218, size: 20.5, color: cArgonIce, align: .center)

        drawFlowingLine(ctx, x1: 405, y1: 250, x2: 270, y2: 340, color: cArgonElectric, t: t, dots: 1)
        drawKeynoteCard(ctx, x: 120, y: 340, w: 300, h: 250, radius: 24, accent: cArgonElectric, headerHeight: 58, focus: 0.5)
        drawText(ctx, "Frozen Backbone W0", x: 270, y: 552, size: 19.5, weight: .bold, color: cArgonIce, align: .center)
        drawMath(ctx, [.i("W"), .sub("0"), .r(" ∈ ℝ"), .sup("768 × 768")], x: 270, y: 488, size: 23, color: cInk, align: .center)

        for gr in 0..<2 {
            for gc in 0..<6 {
                let gx = 172 + CGFloat(gc) * 33
                let gy = 430 + CGFloat(gr) * 19
                drawCard(ctx, x: gx, y: gy, w: 28, h: 14, radius: 4, fill: rgba(59, 140, 255, 0.14), stroke: rgba(124, 194, 255, 0.25), lineWidth: 0.9)
            }
        }
        drawText(ctx, "requires_grad = False", x: 270, y: 394, size: 16.5, weight: .mono, color: cInk, align: .center)
        drawText(ctx, "0 AdamW state bytes", x: 270, y: 362, size: 17.5, weight: .bold, color: cCyan, align: .center)
        drawFlowingLine(ctx, x1: 270, y1: 590, x2: 425, y2: 670, color: cArgonElectric, t: t, dots: 1)

        drawFlowingLine(ctx, x1: 585, y1: 250, x2: 680, y2: 308, color: cViolet, t: t, dots: 2)

        let trA = CGMutablePath()
        trA.move(to: CGPoint(x: 505, y: 308))
        trA.addLine(to: CGPoint(x: 855, y: 308))
        trA.addLine(to: CGPoint(x: 795, y: 406))
        trA.addLine(to: CGPoint(x: 565, y: 406))
        trA.closeSubpath()
        ctx.saveGState(); ctx.addPath(trA); ctx.setFillColor(cVioletTint); ctx.fillPath()
        ctx.addPath(trA); ctx.setStrokeColor(cViolet); ctx.setLineWidth(1.8); ctx.strokePath(); ctx.restoreGState()
        drawMath(ctx, [.i("A"), .r(" ∈ ℝ"), .sup("8 × 768"), .r("   (6,144 params)")], x: 680, y: 366, size: 18.5, color: cViolet, align: .center)
        drawMath(ctx, [.i("A"), .r(" ∼ "), .i("N"), .r("(0, 0.01"), .sup("2"), .r(")")], x: 680, y: 328, size: 17.0, color: cInk, align: .center)

        drawFlowingLine(ctx, x1: 680, y1: 406, x2: 680, y2: 436, color: cAmber, t: t, dots: 1)
        drawFlowingLine(ctx, x1: 680, y1: 476, x2: 680, y2: 504, color: cAmber, t: t + 0.5, dots: 1)

        drawCard(ctx, x: 585, y: 436, w: 190, h: 40, radius: 20, fill: cAmberTint, stroke: cAmber, lineWidth: 1.5)
        for rIdx in 0..<8 {
            let rx: CGFloat = 604 + CGFloat(rIdx) * 21.5
            let p = 0.5 + 0.5 * sin(angle * 2.0 - CGFloat(rIdx) * 0.65)
            let bh: CGFloat = 8 + 18 * p
            drawCard(ctx, x: rx - 4.5, y: 456 - bh / 2, w: 9, h: bh, radius: 4.5, fill: withAlpha(cAmber, 0.55 + 0.45 * p), stroke: nil)
        }
        drawMath(ctx, [.i("z"), .r(" = "), .i("A x")], x: 572, y: 448, size: 18.5, color: cAmber, align: .right)
        drawMath(ctx, [.r("∈ ℝ"), .sup("8"), .r(" (r = 8)")], x: 788, y: 448, size: 18.5, color: cAmber, align: .left)

        let trB = CGMutablePath()
        trB.move(to: CGPoint(x: 565, y: 504))
        trB.addLine(to: CGPoint(x: 795, y: 504))
        trB.addLine(to: CGPoint(x: 855, y: 604))
        trB.addLine(to: CGPoint(x: 505, y: 604))
        trB.closeSubpath()
        ctx.saveGState(); ctx.addPath(trB); ctx.setFillColor(cGreenTint); ctx.fillPath()
        ctx.addPath(trB); ctx.setStrokeColor(cGreen); ctx.setLineWidth(1.8); ctx.strokePath(); ctx.restoreGState()
        drawMath(ctx, [.i("B"), .r(" ∈ ℝ"), .sup("768 × 8"), .r("   (6,144 params)")], x: 680, y: 568, size: 18.5, color: cGreen, align: .center)
        drawMath(ctx, [.i("B"), .r(" = 0  at step 0   ·   "), .frac(num: [.i("α")], den: [.i("r")]), .r(" = 2.0")], x: 680, y: 524, size: 17.5, color: cInk, align: .center)

        drawFlowingLine(ctx, x1: 680, y1: 604, x2: 565, y2: 670, color: cGreen, t: t, dots: 2)

        drawRadialGlow(ctx, cx: 490, cy: 695, radius: 75 + 20 * sin(angle), color: cCyan, alpha: 0.22)
        drawCard(ctx, x: 295, y: 670, w: 390, h: 50, radius: 25, fill: cSurfaceHi, stroke: cCyan, lineWidth: 1.6)
        drawMath(ctx, [.r("⊕   "), .i("h"), .r(" = "), .i("W"), .sub("0"), .i("x"), .r(" + 2.0 · "), .i("B"), .r("("), .i("A x"), .r(")")], x: 490, y: 688, size: 20.5, color: cInk, align: .center)

        drawKeynoteCard(ctx, x: 892, y: 185, w: 442, h: 615, radius: 34, accent: cArgonElectric, headerHeight: 135, focus: columnFocus(index: 0, count: 2, t: t))
        drawText(ctx, "Full Fine-Tuning", x: 1113, y: 742, size: 28, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "67.03M Params · 0.8270 Test F1 (10 ep)", x: 1113, y: 702, size: 19, weight: .bold, color: cArgonIce, align: .center)

        drawText(ctx, "Updates every weight matrix across all", x: 920, y: 615, size: 18.5, weight: .medium, color: cInk)
        drawText(ctx, "6 DistilBERT layers + classifier head:", x: 920, y: 588, size: 18.5, weight: .medium, color: cInk)

        drawFeatureRow(ctx, icon: .layers, iconColor: cArgonElectric, spans: [
            ("Backbone ", .medium, cSlate), ("66.36M", .bold, cArgonIce), (" + Head ", .medium, cSlate), ("0.67M", .bold, cInk)
        ], x: 920, y: 522, size: 17.5)
        drawFeatureRow(ctx, icon: .sparkle, iconColor: cArgonElectric, spans: [
            ("Val LR grid: ", .medium, cSlate), ("5e-5 chosen (0.7909 @ 3ep)", .bold, cArgonIce)
        ], x: 920, y: 468, size: 17.5)
        drawFeatureRow(ctx, icon: .check, iconColor: cGreen, spans: [
            ("3-ep Test ", .medium, cSlate), ("0.7836 F1", .bold, cInk), (" → 10-ep ", .medium, cSlate), ("0.8270 F1", .bold, cGreen)
        ], x: 920, y: 414, size: 17.5)
        drawFeatureRow(ctx, icon: .sliders, iconColor: cInk, spans: [
            ("Peak GPU memory ", .medium, cSlate), ("~4.4 GB", .bold, cInk), (" (11.6m A100)", .medium, cSlate)
        ], x: 920, y: 360, size: 17.5)
        drawFeatureRow(ctx, icon: .bolt, iconColor: cCoral, spans: [
            ("Checkpoint ", .medium, cSlate), ("260 MB per seed", .bold, cCoral), (" (on Drive)", .medium, cSlate)
        ], x: 920, y: 306, size: 17.5)

        drawText(ctx, "*Production accuracy winner (87.98% Test Acc)", x: 1113, y: 218, size: 14.5, weight: .regular, color: cMuted, align: .center)

        drawKeynoteCard(ctx, x: 1362, y: 185, w: 442, h: 615, radius: 34, accent: cGreen, headerHeight: 135, focus: columnFocus(index: 1, count: 2, t: t))
        drawText(ctx, "LoRA Adaptation", x: 1583, y: 742, size: 28, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "0.81M Params (1.2%) · 0.8144 Test F1", x: 1583, y: 702, size: 19, weight: .bold, color: cGreen, align: .center)

        drawText(ctx, "Freezes W0 and trains rank-8 adapters", x: 1390, y: 615, size: 18.5, weight: .medium, color: cInk)
        drawText(ctx, "on attention Q & V projections:", x: 1390, y: 588, size: 18.5, weight: .medium, color: cInk)

        drawFeatureRow(ctx, icon: .sparkle, iconColor: cGreen, spans: [
            ("147,456 ", .bold, cGreen), ("LoRA (q,v) + ", .medium, cSlate), ("667,492 ", .bold, cInk), ("head", .medium, cSlate)
        ], x: 1390, y: 522, size: 17.5)
        drawFeatureRow(ctx, icon: .sliders, iconColor: cInk, spans: [
            ("Val LR grid: ", .medium, cSlate), ("1e-3 chosen (0.7849 @ 3ep)", .bold, cGreen)
        ], x: 1390, y: 468, size: 17.5)
        drawFeatureRow(ctx, icon: .check, iconColor: cCyan, spans: [
            ("3-ep ", .medium, cSlate), ("0.7816", .bold, cInk), (" (all ", .medium, cSlate), ("0.7867", .bold, cCyan), (") → 10-ep ", .medium, cSlate), ("0.8144", .bold, cGreen)
        ], x: 1390, y: 414, size: 17.5)
        drawFeatureRow(ctx, icon: .bolt, iconColor: cGreen, spans: [
            ("4× less VRAM (", .bold, cGreen), ("~1.1 GB", .bold, cInk), (") · ~3 MB in git", .medium, cSlate)
        ], x: 1390, y: 360, size: 17.5)
        drawFeatureRow(ctx, icon: .target, iconColor: cCyan, spans: [
            ("Merges ", .medium, cSlate), ("W0 + 2 B A", .bold, cCyan), (" with 0 ms latency", .medium, cSlate)
        ], x: 1390, y: 306, size: 17.5)

        drawText(ctx, "*Verified to exact param count vs HF peft", x: 1583, y: 218, size: 14.5, weight: .regular, color: cMuted, align: .center)

    // ------------------------------------------------------------------------
    // SLIDE 10: PRE-REGISTERED HYPOTHESIS TESTS & 10-EPOCH FOLLOW-UP
    // ------------------------------------------------------------------------
    case 10:
        drawStageCornerGlows(ctx, topLeftColor: cGreen, bottomRightColor: cArgonElectric, t: t, intensity: 0.16)
        drawTwoToneTitle(ctx, leftText: "Pre-registered tests ", rightText: "& 10-epoch follow-up", rightColor: cGreen)
        drawText(ctx, "Every model comparison on the 10,000 test clauses is decided by paired bootstrap (1,000 resamples) and McNemar's test", x: 960, y: 882, size: 24, weight: .medium, color: cSlate, align: .center)

        let comparisons: [(name: String, sub: String, r1: [(String, FontWeight, CGColor)], r2: [(String, FontWeight, CGColor)], foot: String, col: CGColor)] = [
            ("Registered Test (3 Epochs)", "Full FT 0.7781 vs LoRA 0.7826 (Seed 0)",
             [("Δ Macro-F1 = ", .medium, cSlate), ("-0.0045 [-0.0147, +0.0098]", .bold, cInk)],
             [("McNemar b=323, c=249, ", .medium, cSlate), ("p = 0.002 (all 3 seeds)", .bold, cArgonIce)],
             "*3-seed Δ = +0.0019 [-0.0054, +0.0130]", cArgonElectric),

            ("Follow-Up (10 Epochs)", "Full FT 0.8259 vs LoRA 0.8144 (Seed 0)",
             [("Δ Macro-F1 = ", .medium, cSlate), ("+0.0114 [+0.0020, +0.0200]", .bold, cGreen)],
             [("3-seed Δ = ", .medium, cSlate), ("+0.0126 [+0.0054, +0.0193]", .bold, cInk)],
             "*0.8270 ± 0.0036 vs 0.8144 ± 0.0005 (+13.6 pts Tail)", cGreen),

            ("DistilBERT (10ep) vs TF-IDF", "0.8259 vs 0.8291 on Held-Out Test",
             [("Δ Macro-F1 = ", .medium, cSlate), ("-0.0032 [-0.0128, +0.0067]", .bold, cAmber)],
             [("McNemar b=366, c=351, ", .medium, cSlate), ("p = 0.60 (statistical tie)", .bold, cAmber)],
             "*3-seed Δ = -0.0021 [-0.0102, +0.0060] · CA +18.5 pts", cAmber)
        ]

        let bw: CGFloat = 540
        let bgap: CGFloat = 40
        for (i, b) in comparisons.enumerated() {
            let f = columnFocus(index: i, count: 3, t: t)
            let bx = 110 + CGFloat(i) * (bw + bgap)
            let bcx = bx + bw / 2.0
            drawKeynoteCard(ctx, x: bx, y: 485, w: bw, h: 325, radius: 30, accent: b.col, headerHeight: 95, focus: f)
            drawText(ctx, b.name, x: bcx, y: 760, size: 26, weight: .bold, color: cInk, align: .center)
            drawText(ctx, b.sub, x: bcx, y: 724, size: 18.5, weight: .bold, color: b.col, align: .center)

            drawFeatureRow(ctx, icon: .sparkle, iconColor: b.col, spans: b.r1, x: bx + 28, y: 648, size: 18.0)
            drawFeatureRow(ctx, icon: .check, iconColor: cInk, spans: b.r2, x: bx + 28, y: 590, size: 18.0)

            drawText(ctx, b.foot, x: bcx, y: 525, size: 16.5, weight: .medium, color: cSlate, align: .center)
        }

        drawText(ctx, "Paired Statistical Testing Protocol (src/ledgar/stats.py · notebooks/06_evaluation.ipynb)", x: 140, y: 420, size: 22.5, weight: .bold, color: cCyan)

        drawMath(ctx, [
            .r("1. Continuity-Corrected McNemar Test :     "),
            .i("χ"), .sup("2"), .r(" = "),
            .frac(num: [.r("(|"), .i("b"), .r(" - "), .i("c"), .r("| - 1)"), .sup("2")], den: [.i("b"), .r(" + "), .i("c")]),
            .space(18),
            .r("(per-clause discordant pairs "), .i("b, c"), .r(" on "), .i("N"), .r(" = 10,000)")
        ], x: 140, y: 340, size: 21.0, color: cInk)

        drawMath(ctx, [
            .r("2. 1,000-Iteration Paired Bootstrap 95% CI :     "),
            .r("Δ"), .op("F1"), .sup("(b)"), .r(" = "), .op("F1"), .r("("), .i("M"), .sub("A"), .r(", "), .i("D"), .sup("*(b)"), .r(") - "), .op("F1"), .r("("), .i("M"), .sub("B"), .r(", "), .i("D"), .sup("*(b)"), .r(")")
        ], x: 140, y: 258, size: 21.0, color: cInk)

        drawFeatureRow(ctx, icon: .target, iconColor: cAmber, spans: [
            ("Why English DistilBERT over CamemBERT: ", .bold, cAmber),
            ("LEDGAR is 100% U.S. SEC English; French SentencePiece fragments idioms.", .medium, cInk)
        ], x: 140, y: 188, size: 18.0)

        let bx0: CGFloat = 1220
        let by0: CGFloat = 210
        let bw0: CGFloat = 560
        let bh0: CGFloat = 180
        drawDivider(ctx, x1: bx0, y1: by0, x2: bx0 + bw0, y2: by0, color: cSlate, width: 1.8)

        let gaussArea = CGMutablePath()
        let gaussLine = CGMutablePath()
        let gMid = bx0 + bw0 * 0.56
        let gSig: CGFloat = 82
        let ciLow = gMid - 1.96 * gSig * 0.65
        let ciHigh = gMid + 1.96 * gSig * 0.65

        let sampleX = ciLow + (ciHigh - ciLow) * (0.5 + 0.45 * sin(angle))
        let sampleZ = (sampleX - gMid) / gSig
        let sampleY = by0 + exp(-0.5 * sampleZ * sampleZ) * (bh0 - 25)
        let bIter = 1 + (Int(round(t * 999.0)) % 1000)

        for bBar in 0..<18 {
            let uBar = CGFloat(bBar) / 17.0
            let barX = ciLow + uBar * (ciHigh - ciLow)
            let zBar = (barX - gMid) / gSig
            let barH = exp(-0.5 * zBar * zBar) * (bh0 - 28)
            let isNear = abs(barX - sampleX) < 16
            drawCard(ctx, x: barX - 3.5, y: by0, w: 7, h: barH, radius: 3, fill: withAlpha(isNear ? cCyan : cGreen, isNear ? 0.55 : 0.14), stroke: nil)
        }

        gaussArea.move(to: CGPoint(x: bx0 + 30, y: by0))
        for step in 0...90 {
            let u = CGFloat(step) / 90.0
            let px = bx0 + 30 + u * (bw0 - 60)
            let z = (px - gMid) / gSig
            let py = by0 + exp(-0.5 * z * z) * (bh0 - 25)
            gaussArea.addLine(to: CGPoint(x: px, y: py))
            if step == 0 { gaussLine.move(to: CGPoint(x: px, y: py)) }
            else { gaussLine.addLine(to: CGPoint(x: px, y: py)) }
        }
        gaussArea.addLine(to: CGPoint(x: bx0 + bw0 - 30, y: by0))
        gaussArea.closeSubpath()

        ctx.saveGState()
        ctx.addPath(gaussArea)
        ctx.setFillColor(withAlpha(cGreen, 0.14))
        ctx.fillPath()
        ctx.addPath(gaussLine)
        ctx.setStrokeColor(cGreen)
        ctx.setLineWidth(3.0)
        ctx.strokePath()
        ctx.restoreGState()

        drawDivider(ctx, x1: sampleX, y1: by0, x2: sampleX, y2: sampleY, color: cAmber, width: 2.2)
        ctx.saveGState()
        ctx.setFillColor(cAmber)
        ctx.fillEllipse(in: CGRect(x: sampleX - 6, y: sampleY - 6, width: 12, height: 12))
        ctx.restoreGState()
        drawText(ctx, "b = \(bIter)/1000", x: sampleX, y: sampleY + 16, size: 16.5, weight: .mono, color: cAmber, align: .center)

        let zeroX = bx0 + bw0 * 0.18
        drawDivider(ctx, x1: zeroX, y1: by0, x2: zeroX, y2: by0 + bh0 - 15, color: cCoral, width: 1.8, dashed: true)
        drawText(ctx, "Δ = 0", x: zeroX, y: by0 - 30, size: 18.5, weight: .bold, color: cCoral, align: .center)

        drawDivider(ctx, x1: ciLow, y1: by0, x2: ciLow, y2: by0 + bh0 * 0.45, color: cCyan, width: 1.6, dashed: true)
        drawDivider(ctx, x1: ciHigh, y1: by0, x2: ciHigh, y2: by0 + bh0 * 0.45, color: cCyan, width: 1.6, dashed: true)
        drawText(ctx, "10-ep Full vs LoRA: +1.14 pts [+0.20, +2.00]", x: gMid + 24, y: by0 - 30, size: 18.0, weight: .bold, color: cGreen, align: .center)
        drawMath(ctx, [.i("P"), .r("(Δ"), .op("Macro-F1"), .r(")")], x: gMid, y: by0 + bh0 + 24, size: 19.5, color: cGreen, align: .center)

    // ------------------------------------------------------------------------
    // SLIDE 11: SEMANTIC PRECEDENT RETRIEVAL ON UNIT HYPERSPHERE (m2)
    // ------------------------------------------------------------------------
    case 11:
        drawStageCornerGlows(ctx, topLeftColor: cCyan, bottomRightColor: cGreen, t: t, intensity: 0.18)
        drawTwoToneTitle(ctx, leftText: "Precedent retrieval: ", rightText: "78 candidates & within-type gate", rightColor: cCyan)
        drawText(ctx, "Evaluating 13 bank encodings × 6 hybrid variants on 60,000 clauses (notebooks/07_retrieval.ipynb)", x: 960, y: 882, size: 24, weight: .medium, color: cSlate, align: .center)

        drawMath(ctx, [
            .op("score"), .r("("), .i("q, d"), .r(") = "),
            .b("q", cCyan), .sup("⊤"), .b("d", cGreen),
            .r(" + "), .i("λ"), .r(" "), .op("log"), .r(" "), .i("p"), .r("("), .i("y"), .sub("d"), .r(" | "), .i("q"), .r(")     where   "),
            .b("q", cCyan), .r(", "), .b("d", cGreen), .r(" ∈ "), .i("S"), .sup("d-1")
        ], x: 485, y: 772, size: 24.0, color: cInk, align: .center)

        let ox: CGFloat = 185
        let oy: CGFloat = 235
        let rad: CGFloat = 420

        drawVectorArrow(ctx, x1: ox, y1: oy, x2: ox + rad + 45, y2: oy, color: cSlate, width: 2.2, headLength: 10)
        drawVectorArrow(ctx, x1: ox, y1: oy, x2: ox, y2: oy + rad + 45, color: cSlate, width: 2.2, headLength: 10)

        ctx.saveGState()
        ctx.setStrokeColor(withAlpha(cArgonElectric, 0.50))
        ctx.setLineWidth(2.4)
        ctx.setLineDash(phase: 0, lengths: [8, 6])
        ctx.addArc(center: CGPoint(x: ox, y: oy), radius: rad, startAngle: 0, endAngle: .pi / 2.0, clockwise: false)
        ctx.strokePath()
        ctx.restoreGState()
        drawMath(ctx, [.r("||"), .b("u", cArgonIce), .r("||"), .sub("2"), .r(" = 1  (Centring fixes MLM anisotropy 0.88 → 0.00)")], x: ox + 48, y: oy + 22, size: 16.5, color: cArgonIce)

        let qAng: CGFloat = 36.0 * .pi / 180.0
        let qx = ox + rad * cos(qAng)
        let qy = oy + rad * sin(qAng)

        let d1Ang: CGFloat = 52.0 * .pi / 180.0
        let d1x = ox + rad * cos(d1Ang)
        let d1y = oy + rad * sin(d1Ang)

        let d2Ang: CGFloat = 23.0 * .pi / 180.0
        let d2x = ox + rad * cos(d2Ang)
        let d2y = oy + rad * sin(d2Ang)

        let dnAng: CGFloat = 82.0 * .pi / 180.0
        let dnx = ox + rad * cos(dnAng)
        let dny = oy + rad * sin(dnAng)

        let conePath = CGMutablePath()
        conePath.move(to: CGPoint(x: ox, y: oy))
        conePath.addArc(center: CGPoint(x: ox, y: oy), radius: rad, startAngle: d2Ang, endAngle: d1Ang, clockwise: false)
        conePath.closeSubpath()
        ctx.saveGState()
        ctx.addPath(conePath)
        ctx.setFillColor(withAlpha(cGreen, 0.12 + 0.04 * sin(angle)))
        ctx.fillPath()
        ctx.restoreGState()

        let sweepAng = d2Ang + (d1Ang - d2Ang) * (0.5 + 0.5 * sin(angle))
        let sweepX = ox + rad * cos(sweepAng)
        let sweepY = oy + rad * sin(sweepAng)
        drawDivider(ctx, x1: ox, y1: oy, x2: sweepX, y2: sweepY, color: withAlpha(cArgonIce, 0.55), width: 1.8)
        ctx.saveGState()
        ctx.setFillColor(cArgonIce)
        ctx.fillEllipse(in: CGRect(x: sweepX - 5, y: sweepY - 5, width: 10, height: 10))
        ctx.restoreGState()

        let projLen = rad * cos(d1Ang - qAng)
        let projX = ox + projLen * cos(qAng)
        let projY = oy + projLen * sin(qAng)
        drawDivider(ctx, x1: d1x, y1: d1y, x2: projX, y2: projY, color: cAmber, width: 2.2, dashed: true)
        drawDivider(ctx, x1: ox, y1: oy, x2: projX, y2: projY, color: withAlpha(cAmber, 0.55 + 0.35 * sin(angle)), width: 6.5)

        drawVectorArrow(ctx, x1: ox, y1: oy, x2: dnx, y2: dny, color: withAlpha(cCoral, 0.70), width: 2.4, headLength: 10)
        drawVectorArrow(ctx, x1: ox, y1: oy, x2: d1x, y2: d1y, color: cGreen, width: 3.0, headLength: 12)
        drawVectorArrow(ctx, x1: ox, y1: oy, x2: d2x, y2: d2y, color: cGreen, width: 2.6, headLength: 11)
        drawVectorArrow(ctx, x1: ox, y1: oy, x2: qx, y2: qy, color: cCyan, width: 3.8, headLength: 14)

        ctx.saveGState()
        ctx.setStrokeColor(cAmber)
        ctx.setLineWidth(2.4)
        ctx.addArc(center: CGPoint(x: ox, y: oy), radius: 155, startAngle: qAng, endAngle: d1Ang, clockwise: false)
        ctx.strokePath()
        ctx.restoreGState()
        drawMath(ctx, [.i("θ"), .r(" = 19.6°")], x: ox + 108, y: oy + 148, size: 18.5, color: cAmber)

        drawMath(ctx, [.b("q", cCyan), .r(" : "), .op("Query (\"30-day notice\")")], x: qx + 12, y: qy - 6, size: 18.5, color: cCyan)
        drawMath(ctx, [.b("d", cGreen), .sub("1"), .r(" : "), .op("Precedent #1 (cos θ = 0.942)")], x: d1x + 12, y: d1y + 8, size: 18.5, color: cGreen)
        drawMath(ctx, [.b("d", cGreen), .sub("2"), .r(" : "), .op("Precedent #2 (cos θ = 0.918)")], x: d2x + 12, y: d2y - 10, size: 18.5, color: cGreen)
        drawMath(ctx, [.b("d", cCoral), .sub("⊥"), .r(" : "), .op("Base Salary (cos θ = 0.18)")], x: dnx + 12, y: dny + 12, size: 18.0, color: cCoral)

        drawKeynoteCard(ctx, x: 935, y: 500, w: 865, h: 315, radius: 30, accent: cCyan, headerHeight: 88, focus: columnFocus(index: 0, count: 2, t: t))
        drawText(ctx, "All-Pooled vs Semantic-Separated Winner", x: 1367.5, y: 768, size: 26, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "Gate: Perturbation MRR ≥ 0.95 × max (0.852 all-pooled · 0.705 semantic)", x: 1367.5, y: 736, size: 18.5, weight: .medium, color: cCyan, align: .center)

        drawFeatureRow(ctx, icon: .sparkle, iconColor: cCyan, spans: [
            ("BGE-small Centred (Semantic Winner): ", .bold, cCyan), ("Val P@10 = 0.669 · +0.76 pts vs TF-IDF", .medium, cInk)
        ], x: 968, y: 668, size: 18.0)
        drawFeatureRow(ctx, icon: .check, iconColor: cGreen, spans: [
            ("Low-Lexical-Overlap Quartile (n=2,502): ", .bold, cGreen), ("BGE +1.85 pts macro / +4.64 pts micro!", .medium, cInk)
        ], x: 968, y: 616, size: 18.0)
        drawFeatureRow(ctx, icon: .layers, iconColor: cArgonElectric, spans: [
            ("TF-IDF Dense (All-Pooled Winner): ", .bold, cArgonIce), ("Perturb MRR 0.897 · Test P@10 = 0.658", .medium, cInk)
        ], x: 968, y: 564, size: 18.0)
        drawText(ctx, "*Test P@10 by band: Head 0.83 · Middle 0.70 · Tail 0.43 · Legal-BERT Centred P@10 = 0.674", x: 1367.5, y: 518, size: 15.0, weight: .regular, color: cMuted, align: .center)

        drawKeynoteCard(ctx, x: 935, y: 155, w: 865, h: 315, radius: 30, accent: cGreen, headerHeight: 88, focus: columnFocus(index: 1, count: 2, t: t))
        drawText(ctx, "Why Hybrid Collapses Within-Type & Where S-BERT Wins", x: 1367.5, y: 423, size: 25.5, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "Pasted Clause vs Natural-Language Lawyer Question", x: 1367.5, y: 391, size: 18.5, weight: .medium, color: cGreen, align: .center)

        drawFeatureRow(ctx, icon: .cross, iconColor: cCoral, spans: [
            ("LoRA Features & Hybrids: ", .bold, cCoral), ("Win type proxy (0.758 / 0.805 P@10) but collapse", .medium, cInk)
        ], x: 968, y: 328, size: 18.0)
        drawText(ctx, "within-type Perturb MRR to 0.508 / 0.40-0.78 → keep type as a UI checkbox!", x: 998, y: 298, size: 17.5, weight: .bold, color: cSlate)

        drawFeatureRow(ctx, icon: .bolt, iconColor: cGreen, spans: [
            ("Lawyer NL Questions (\"30-day notice\"): ", .bold, cGreen), ("E5 & MiniLM reach 86-88% constraint", .medium, cInk)
        ], x: 968, y: 246, size: 18.0)
        drawText(ctx, "precision (4× lift over base rate) & 0.56-0.59 type-name P@10 (vs 0.48 TF-IDF).", x: 998, y: 216, size: 17.5, weight: .medium, color: cSlate)
        drawText(ctx, "*TF-IDF weakness: hubness skew = 4.71 (one boilerplate clause in 414 top-10 lists)", x: 1367.5, y: 172, size: 15.0, weight: .regular, color: cMuted, align: .center)

    // ------------------------------------------------------------------------
    // SLIDE 12: BIAS & FAILURE-MODE AUDIT PART 1 (TAIL RARITY, COPIES & TRUNCATION)
    // ------------------------------------------------------------------------
    case 12:
        drawStageCornerGlows(ctx, topLeftColor: cCoral, bottomRightColor: cAmber, t: t, intensity: 0.17)
        drawTwoToneTitle(ctx, leftText: "Audit I: ", rightText: "tail rarity, copies & truncation", rightColor: cCoral)
        drawText(ctx, "Held-out test evaluation (10,000 clauses) by training support band, verbatim overlap, and >512 length", x: 960, y: 882, size: 24, weight: .medium, color: cSlate, align: .center)

        drawText(ctx, "Failure Mode 1 · Zipfian Tail Degradation (Test Set, Seed 0)", x: 150, y: 780, size: 19, weight: .bold, color: cCoral)
        drawText(ctx, "-19.7 pts Macro-F1 drop from Head (0.958) to Tail (0.761)", x: 150, y: 736, size: 27, weight: .bold, color: cInk)
        drawText(ctx, "10-Epoch Tail Gain: 0.625 (3 ep) → 0.761 (10 ep) (+13.6 pts) · 26.8% still missed!", x: 150, y: 690, size: 18.5, weight: .bold, color: cCoral)

        let buckets: [(name: String, subMath: [MathToken], f1: CGFloat, rec: CGFloat, col: CGColor)] = [
            ("Head (16 types)", [.i("N"), .sub("c"), .r(" ≥ 1,000 (n=4,166)")], 0.958, 0.962, cArgonElectric),
            ("Middle (62 types)", [.r("300 ≤ "), .i("N"), .sub("c"), .r(" < 1,000 (n=5,183)")], 0.869, 0.865, cAmber),
            ("Tail (22 types)", [.i("N"), .sub("c"), .r(" < 300 (n=651)")], 0.761, 0.732, cCoral)
        ]

        let baseBarY: CGFloat = 250
        let maxBarH: CGFloat  = 340
        drawDivider(ctx, x1: 150, y1: baseBarY, x2: 880, y2: baseBarY, color: cSlate, width: 1.8)

        var recCenters: [CGPoint] = []
        for (i, b) in buckets.enumerated() {
            let bx: CGFloat = 195 + CGFloat(i) * 240
            let hF1  = maxBarH * b.f1
            let hRec = maxBarH * b.rec

            drawCard(ctx, x: bx, y: baseBarY, w: 76, h: hF1, radius: 10, fill: b.col, stroke: nil)
            drawText(ctx, String(format: "%.3f", b.f1), x: bx + 38, y: baseBarY + hF1 + 12, size: 20, weight: .bold, color: b.col, align: .center)
            drawText(ctx, "F1", x: bx + 38, y: baseBarY + 14, size: 16.5, weight: .bold, color: cBg, align: .center)

            drawCard(ctx, x: bx + 84, y: baseBarY, w: 76, h: hRec, radius: 10, fill: withAlpha(b.col, 0.30), stroke: b.col, lineWidth: 1.5)
            drawText(ctx, String(format: "%.3f", b.rec), x: bx + 122, y: baseBarY + hRec + 12, size: 20, weight: .bold, color: cInk, align: .center)
            drawText(ctx, "Rec", x: bx + 122, y: baseBarY + 14, size: 16.5, weight: .bold, color: cInk, align: .center)

            recCenters.append(CGPoint(x: bx + 122, y: baseBarY + hRec))

            if i == 2 {
                let missH = maxBarH * (1.0 - b.rec)
                let pulseAlpha: CGFloat = 0.16 + 0.10 * (0.5 + 0.5 * sin(angle * 2.0))
                drawCard(ctx, x: bx + 84, y: baseBarY + hRec + 34, w: 76, h: missH - 34, radius: 10, fill: withAlpha(cCoral, pulseAlpha), stroke: withAlpha(cCoral, 0.70), lineWidth: 1.5)
                drawText(ctx, "26.8%", x: bx + 122, y: baseBarY + hRec + 64, size: 17, weight: .bold, color: cInk, align: .center)
                drawText(ctx, "MISSED", x: bx + 122, y: baseBarY + hRec + 44, size: 13.5, weight: .bold, color: cCoral, align: .center)
            }

            drawText(ctx, b.name, x: bx + 80, y: baseBarY - 34, size: 19.5, weight: .bold, color: cInk, align: .center)
            drawMath(ctx, b.subMath, x: bx + 80, y: baseBarY - 62, size: 17, color: cSlate, align: .center)
        }

        drawFlowingLine(ctx, x1: recCenters[0].x, y1: recCenters[0].y, x2: recCenters[1].x, y2: recCenters[1].y, color: cAmber, t: t, dots: 1)
        drawFlowingLine(ctx, x1: recCenters[1].x, y1: recCenters[1].y, x2: recCenters[2].x, y2: recCenters[2].y, color: cCoral, t: t + 0.5, dots: 1)

        drawText(ctx, "Types < 0.50 Test F1: Applicable Laws (0.23) · Assigns (0.40, n=4) · Venues (0.47)", x: 150, y: 145, size: 18.0, weight: .bold, color: cInk)

        drawText(ctx, "Failure Mode 2 · Boilerplate Copies & >512 Truncation", x: 1020, y: 780, size: 19, weight: .bold, color: cAmber)
        drawText(ctx, "Verbatim copies score 0.985 F1 vs 0.822 on unseen", x: 1020, y: 736, size: 27, weight: .bold, color: cInk)
        drawText(ctx, "282 test clauses (2.8%) appear word-for-word in train; 175 clauses (1.8%) exceed 512 tokens.", x: 1020, y: 690, size: 18.5, weight: .medium, color: cSlate)

        let truncs: [(cls: String, pct: String, val: CGFloat, col: CGColor)] = [
            ("Seen in Train (n = 282 copies)", "0.985 Macro-F1 (98.9% Acc)", 0.985, cGreen),
            ("Unseen in Train (n = 9,718)", "0.822 Macro-F1 (87.5% Acc)", 0.822, cArgonElectric),
            ("Truncated > 512 Tokens (n = 175)", "0.866 Macro-F1 (end cut off)", 0.866, cAmber),
            ("Capitalization (> 256 / > 512)", "69.0% > 256  ·  16.2% > 512", 0.690, cCoral),
            ("Retrieval Tail (< 300 train)", "0.430 Test P@10 (vs 0.830 Head)", 0.430, cCoral)
        ]

        let activeTrunc = Int(floor(t * 5.0)) % 5
        for (idx, tr) in truncs.enumerated() {
            let ty: CGFloat = 595 - CGFloat(idx) * 76
            let isCur = (idx == activeTrunc)
            drawText(ctx, tr.cls, x: 1020, y: ty + 20, size: 19.5, weight: .bold, color: isCur ? tr.col : cInk)
            drawText(ctx, tr.pct, x: 1790, y: ty + 20, size: 18.5, weight: .bold, color: tr.col, align: .right)
            drawCard(ctx, x: 1020, y: ty - 8, w: 770, h: 16, radius: 8, fill: rgba(255, 255, 255, 0.07), stroke: nil)
            drawCard(ctx, x: 1020, y: ty - 8, w: 770 * tr.val, h: 16, radius: 8, fill: tr.col, stroke: isCur ? cInk : nil, lineWidth: 1.2)
        }

        drawKeynoteCard(ctx, x: 1020, y: 132, w: 770, h: 106, radius: 22, accent: cAmber, headerHeight: nil, focus: 0.6)
        drawText(ctx, "Operational Guardrail: Flag every clause with L > 512 tokens or Tail class (Nc < 300)", x: 1405, y: 196, size: 18.5, weight: .bold, color: cAmber, align: .center)
        drawText(ctx, "and report seen-vs-unseen accuracy separately so verbatim boilerplate never inflates trust.", x: 1405, y: 160, size: 17.5, weight: .medium, color: cInk, align: .center)

    // ------------------------------------------------------------------------
    // SLIDE 13: BIAS & FAILURE-MODE AUDIT PART 2 (CONTRACTS, PARTIES, JURISDICTIONS & ALL-CAPS)
    // ------------------------------------------------------------------------
    case 13:
        drawStageCornerGlows(ctx, topLeftColor: cViolet, bottomRightColor: cCyan, t: t, intensity: 0.17)
        drawTwoToneTitle(ctx, leftText: "Audit II: ", rightText: "contracts, parties, jurisdictions & ALL-CAPS", rightColor: cViolet)
        drawText(ctx, "Quantitative bias audit across 10,000 test clauses (src/ledgar/data.py · reports/bias_audit.md)", x: 960, y: 882, size: 24, weight: .medium, color: cSlate, align: .center)

        drawText(ctx, "Failure Mode 3 · SEC Domain & Party-Role Representation Bias", x: 130, y: 780, size: 19, weight: .bold, color: cViolet)
        drawText(ctx, "Leases & Landlord/Tenant drop -14 to -17 pts in accuracy", x: 130, y: 736, size: 27, weight: .bold, color: cInk)
        drawText(ctx, "SEC Exhibit-10 filings over-represent loans & M&A (89-91% Acc) vs leases (2.5% of train).", x: 130, y: 692, size: 18.5, weight: .medium, color: cSlate)

        let slices: [(name: String, meta: String, f1: CGFloat, acc: String, col: CGColor)] = [
            ("Borrower / Lender (n = 1,738, 19.7%)", "90.1% Acc · 0.837 F1 [0.809, 0.861]", 0.837, "0.837 F1", cGreen),
            ("Credit / Loan (n = 1,855, 21.0%)", "89.4% Acc · 0.818 F1 [0.788, 0.841]", 0.818, "0.818 F1", cArgonElectric),
            ("M&A / Purchase (n = 637, 8.8%)", "89.2% Acc · 0.814 F1 [0.771, 0.859]", 0.814, "0.814 F1", cCyan),
            ("Employment / Equity (n = 2,106, 18.2%)", "86.1% Acc · 0.732 F1 [0.701, 0.767]", 0.732, "0.732 F1", cAmber),
            ("Lease / License (n = 281, 2.5%)", "75.8% Acc · 0.657 F1 [0.600, 0.727]", 0.657, "0.657 F1", cCoral),
            ("Landlord / Tenant (n = 188, 1.6%)", "73.4% Acc · 0.634 F1 [0.573, 0.712]", 0.634, "0.634 F1", cCoral)
        ]

        let activeSlice = Int(floor(t * 6.0)) % 6
        for (idx, sl) in slices.enumerated() {
            let sy: CGFloat = 615 - CGFloat(idx) * 78
            let isCur = (idx == activeSlice)
            drawText(ctx, sl.name, x: 130, y: sy + 20, size: 18.5, weight: .bold, color: isCur ? sl.col : cInk)
            drawText(ctx, sl.meta, x: 905, y: sy + 20, size: 17.5, weight: .bold, color: sl.col, align: .right)
            drawCard(ctx, x: 130, y: sy - 6, w: 775, h: 15, radius: 7.5, fill: rgba(255, 255, 255, 0.07), stroke: nil)
            drawCard(ctx, x: 130, y: sy - 6, w: 775 * (sl.f1 / 0.90), h: 15, radius: 7.5, fill: sl.col, stroke: isCur ? cInk : nil, lineWidth: 1.2)
        }

        drawKeynoteCard(ctx, x: 940, y: 495, w: 865, h: 315, radius: 30, accent: cCyan, headerHeight: 88, focus: columnFocus(index: 0, count: 2, t: t))
        drawText(ctx, "Jurisdictional & ALL-CAPS Typography Bias", x: 1372.5, y: 765, size: 25.5, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "California (+18.5 pts for DistilBERT) & Uppercase Collapse (0.517 F1)", x: 1372.5, y: 733, size: 18.5, weight: .medium, color: cCyan, align: .center)

        drawFeatureRow(ctx, icon: .sparkle, iconColor: cGreen, spans: [
            ("California (n=65, 0.9% tr): ", .bold, cGreen), ("DistilBERT 0.723 [0.581, 0.879] vs TF-IDF 0.538", .medium, cInk)
        ], x: 972, y: 662, size: 18.0)
        drawFeatureRow(ctx, icon: .layers, iconColor: cArgonElectric, spans: [
            ("New York (n=358): ", .bold, cArgonIce), ("89.9% Acc (0.801 F1) · ", .medium, cInk), ("Delaware (n=188): ", .bold, cArgonIce), ("88.8% Acc (0.710 F1)", .medium, cInk)
        ], x: 972, y: 610, size: 18.0)
        drawFeatureRow(ctx, icon: .bolt, iconColor: cCoral, spans: [
            ("ALL-CAPS (>50% upper, n=243): ", .bold, cCoral), ("88.5% Acc on Jury Waiver/Warranty, but", .medium, cInk)
        ], x: 972, y: 558, size: 18.0)
        drawText(ctx, "0.517 Macro-F1 [0.454, 0.667] across 29 uppercase types (-30.9 pts vs standard 0.826)!", x: 1002, y: 524, size: 18.0, weight: .bold, color: cCoral)

        drawKeynoteCard(ctx, x: 940, y: 150, w: 865, h: 315, radius: 30, accent: cAmber, headerHeight: 88, focus: columnFocus(index: 1, count: 2, t: t))
        drawText(ctx, "Near-Synonym Confusion & Temperature Calibration", x: 1372.5, y: 420, size: 25.5, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "Post-Hoc Logit Scaling T = 2.47 Cuts Test ECE by 73% (0.086 → 0.023)", x: 1372.5, y: 388, size: 18.5, weight: .medium, color: cAmber, align: .center)

        drawFeatureRow(ctx, icon: .branch, iconColor: cAmber, spans: [
            ("Top Test Confusions: ", .bold, cAmber), ("Applicable ↔ Governing Laws (52), Tax ↔ Withholdings (35),", .medium, cInk)
        ], x: 972, y: 318, size: 18.0)
        drawText(ctx, "Defined Terms ↔ Definitions (33), No Waivers ↔ Waivers (27), Amendments ↔ Modifications (23)", x: 998, y: 286, size: 16.5, weight: .medium, color: cSlate)

        drawFeatureRow(ctx, icon: .shield, iconColor: cCoral, spans: [
            ("ECE & Calibration: ", .bold, cCoral), ("Test ECE rises 0.033 (3ep) → 0.086 (10ep, 57.1% errs > 0.90);", .medium, cInk)
        ], x: 972, y: 228, size: 18.0)
        drawText(ctx, "Fitting T = 2.47 on Val NLL restores Test ECE = 0.023 (5.7% errs > 0.90) with 0 F1 change!", x: 1002, y: 196, size: 18.0, weight: .bold, color: cGreen)

    // ------------------------------------------------------------------------
    // SLIDE 14: OPERATIONAL TRUST BOUNDARY & RED LINES FOR EUROPEAN CLIENTS
    // ------------------------------------------------------------------------
    case 14:
        drawStageCornerGlows(ctx, topLeftColor: cGreen, bottomRightColor: cCoral, t: t, intensity: 0.17)
        drawTwoToneTitle(ctx, leftText: "Where ClauseGuard AI can and ", rightText: "cannot be trusted", rightColor: cGreen)
        drawText(ctx, "Explicit operational contract and model card boundaries (reports/model_card.md) for European legal teams", x: 960, y: 882, size: 24, weight: .medium, color: cSlate, align: .center)

        let fLeft = columnFocus(index: 0, count: 2, t: t)
        drawKeynoteCard(ctx, x: 115, y: 140, w: 815, h: 680, radius: 36, accent: cGreen, headerHeight: 142, focus: fLeft)
        drawText(ctx, "Trusted Operating Zones", x: 522.5, y: 750, size: 32, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "Human-in-the-Loop Assistive Copilot", x: 522.5, y: 708, size: 21, weight: .medium, color: cGreen, align: .center)

        drawText(ctx, "Backed by reproducible empirical evidence across", x: 155, y: 622, size: 21, weight: .bold, color: cInk)
        drawText(ctx, "10,000 held-out test clauses:", x: 155, y: 592, size: 21, weight: .bold, color: cInk)

        drawFeatureRow(ctx, icon: .check, iconColor: cGreen, spans: [
            ("First-pass tagging: ", .bold, cInk), ("Credit/Loan (89.4%), M&A (89.2%) & Securities (91.2%)", .medium, cGreen)
        ], x: 155, y: 520, size: 18.5)
        drawFeatureRow(ctx, icon: .sparkle, iconColor: cCyan, spans: [
            ("Frequent types (≥1,000 train): ", .bold, cInk), ("0.958 Macro-F1 across 4,166 test clauses", .medium, cCyan)
        ], x: 155, y: 462, size: 18.5)
        drawFeatureRow(ctx, icon: .bolt, iconColor: cInk, spans: [
            ("Dual retrieval: ", .bold, cInk), ("BGE+center (clauses, 0.669 P@10) + E5 (NL queries)", .medium, cArgonIce)
        ], x: 155, y: 404, size: 18.5)
        drawFeatureRow(ctx, icon: .layers, iconColor: cGreen, spans: [
            ("Group near-synonym pairs: ", .bold, cInk), ("Governing/Applicable Law & Waivers in UI", .medium, cGreen)
        ], x: 155, y: 346, size: 18.5)
        drawFeatureRow(ctx, icon: .shield, iconColor: cInk, spans: [
            ("Lawyer-ticked type filter: ", .bold, cInk), ("preserves within-type wording similarity", .medium, cCyan)
        ], x: 155, y: 288, size: 18.5)

        drawText(ctx, "*Accelerates junior lawyer triage while keeping human sign-off", x: 522.5, y: 185, size: 15.5, weight: .regular, color: cMuted, align: .center)

        let fRight = columnFocus(index: 1, count: 2, t: t)
        drawKeynoteCard(ctx, x: 990, y: 140, w: 815, h: 680, radius: 36, accent: cCoral, headerHeight: 142, focus: fRight)
        drawText(ctx, "Strict Red Lines", x: 1397.5, y: 750, size: 32, weight: .bold, color: cInk, align: .center)
        drawText(ctx, "Prohibited Autonomous Use Cases", x: 1397.5, y: 708, size: 21, weight: .medium, color: cCoral, align: .center)

        drawText(ctx, "Must never be deployed as an autonomous legal", x: 1030, y: 622, size: 21, weight: .bold, color: cInk)
        drawText(ctx, "decision-maker in the following scenarios:", x: 1030, y: 592, size: 21, weight: .bold, color: cInk)

        drawFeatureRow(ctx, icon: .cross, iconColor: cCoral, spans: [
            ("Autonomous sign-off: ", .bold, cInk), ("labels topic only, never enforceability or caps", .medium, cCoral)
        ], x: 1030, y: 520, size: 18.5)
        drawFeatureRow(ctx, icon: .cross, iconColor: cCoral, spans: [
            ("Leases & IP licenses: ", .bold, cInk), ("75.8% Acc (0.657 F1) & 73.4% on landlord/tenant", .medium, cCoral)
        ], x: 1030, y: 462, size: 18.5)
        drawFeatureRow(ctx, icon: .bolt, iconColor: cAmber, spans: [
            ("ALL-CAPS outside waivers: ", .bold, cInk), ("0.517 Macro-F1 across 29 uppercase types", .medium, cAmber)
        ], x: 1030, y: 404, size: 18.5)
        drawFeatureRow(ctx, icon: .sliders, iconColor: cCoral, spans: [
            ("Uncalibrated p_max: ", .bold, cInk), ("ECE 0.086 at T=1.0 → apply T=2.47 (ECE 0.023)", .medium, cCoral)
        ], x: 1030, y: 346, size: 18.5)
        drawFeatureRow(ctx, icon: .branch, iconColor: cInk, spans: [
            ("Non-US / non-English & L > 512: ", .bold, cInk), ("requires domain/language re-training", .medium, cAmber)
        ], x: 1030, y: 288, size: 18.5)

        drawText(ctx, "*Mandatory European AI governance boundary for CTO sign-off", x: 1397.5, y: 185, size: 15.5, weight: .regular, color: cMuted, align: .center)

    // ------------------------------------------------------------------------
    // SLIDE 15: GEMINI 4 ARGON DARK STAGE FINALE ("Thank you", 15.0s CYCLE)
    // ------------------------------------------------------------------------
    case 15:
        drawStageCornerGlows(ctx, topLeftColor: cArgonElectric, bottomRightColor: cGreen, t: t, intensity: 0.24)

        let finTop = "REPRODUCIBLE PIPELINE   ·   DISTILBERT + LORA   ·   COSINE RETRIEVAL"
        let finBot = "OFFICIAL LEXGLUE   ·   1,000 BOOTSTRAP CI   ·   TEMPERATURE CALIBRATION"
        let drift = 46.0 * sin(angle)
        drawText(ctx, finTop, x: 960 + drift, y: 785, size: 122, weight: .bold, color: withAlpha(cInk, 0.048), align: .center, tracking: -2.8)
        drawText(ctx, finBot, x: 960 - drift, y: 165, size: 122, weight: .bold, color: withAlpha(cArgonElectric, 0.055), align: .center, tracking: -2.8)

        drawTwoToneTitle(ctx, leftText: "Thank ", rightText: "you", rightColor: cArgonElectric, y: 560, size: 116, centerX: 960, tracking: -3.0)

        let clipRect = CGRect(x: 160, y: 425, width: 1600, height: 78)
        ctx.saveGState()
        ctx.clip(to: clipRect)

        let finPhrases: [(left: String, right: String, col: CGColor)] = [
            ("Rigorous · ", "DistilBERT 10-ep (0.827 F1) vs TF-IDF (0.829 F1, McNemar p=0.60) & LoRA (0.814 F1)", cArgonElectric),
            ("Diagnostic · ", "78 retrieval candidates: BGE+center wins semantic P@10 (0.669), E5 wins NL queries", cGreen),
            ("Audited · ", "Contract-kind, party-role, jurisdiction & ALL-CAPS bias slices + explicit EU red lines", cAmber),
            ("Rigorous · ", "DistilBERT 10-ep (0.827 F1) vs TF-IDF (0.829 F1, McNemar p=0.60) & LoRA (0.814 F1)", cArgonElectric)
        ]

        let scaled = t * 3.0
        let seg = Int(floor(scaled))
        let frac = scaled - CGFloat(seg)
        let glide: CGFloat
        if frac < 0.85 {
            glide = 0.0
        } else {
            let u = (frac - 0.85) / 0.15
            glide = u * u * (3.0 - 2.0 * u)
        }
        let scrollPos = CGFloat(seg) + glide
        let slotH: CGFloat = 78.0

        for (idx, item) in finPhrases.enumerated() {
            let itemY = 446.0 + (CGFloat(idx) - scrollPos) * slotH
            if itemY > 345 && itemY < 545 {
                let dist = abs(CGFloat(idx) - scrollPos)
                let alpha = max(0.0, 1.0 - dist * 1.15)
                let w1 = measureText(item.left, size: 36, weight: .bold, tracking: -0.8).width
                let w2 = measureText(item.right, size: 36, weight: .medium, tracking: -0.6).width
                let sx = 960 - (w1 + w2) / 2.0
                drawText(ctx, item.left, x: sx, y: itemY, size: 36, weight: .bold, color: withAlpha(item.col, alpha), tracking: -0.8)
                drawText(ctx, item.right, x: sx + w1, y: itemY, size: 36, weight: .medium, color: withAlpha(cInk, alpha), tracking: -0.6)
            }
        }
        ctx.restoreGState()

        drawText(ctx, "Jonathan   ·   Keira   ·   Alexis", x: 960, y: 300, size: 28, weight: .bold, color: cInk, align: .center, tracking: 1.5)

    default:
        break
    }

    return ctx.makeImage()!
}

// ============================================================================
// HIGH-DEFINITION 256-COLOR PALETTE QUANTIZER FOR CRYSTAL-CLEAR 1080P GIFS
// ============================================================================
let customPaletteRGB: [(UInt8, UInt8, UInt8)] = {
    var pal: [(UInt8, UInt8, UInt8)] = []
    for i in 0..<32 {
        let u = Double(i) / 31.0
        let v = UInt8(round(pow(u, 1.45) * 250.0))
        pal.append((v, v, v))
    }
    for i in 0..<32 {
        let f = pow(Double(i) / 31.0, 1.45)
        let r = UInt8(round(1.0 + f * 65.0))
        let g = UInt8(round(3.0 + f * 135.0))
        let b = UInt8(round(8.0 + f * 245.0))
        pal.append((r, g, b))
    }
    let accents: [(Double, Double, Double)] = [
        (59, 140, 255),
        (78, 205, 196),
        (68, 186, 101),
        (251, 188, 4),
        (255, 107, 107),
        (175, 92, 247)
    ]
    for (ar, ag, ab) in accents {
        for i in 0..<32 {
            let u = Double(i) / 31.0
            let r: Double
            let g: Double
            let b: Double
            if u <= 0.84 {
                let k = pow(u / 0.84, 1.65)
                r = ar * k
                g = ag * k
                b = ab * k
            } else {
                let k = (u - 0.84) / 0.16
                r = ar + (248.0 - ar) * k
                g = ag + (249.0 - ag) * k
                b = ab + (250.0 - ab) * k
            }
            pal.append((
                UInt8(clamping: Int(round(r))),
                UInt8(clamping: Int(round(g))),
                UInt8(clamping: Int(round(b)))
            ))
        }
    }
    while pal.count < 256 { pal.append((0, 0, 0)) }
    return Array(pal.prefix(256))
}()

let paletteFlatBytes: [UInt8] = {
    var bytes: [UInt8] = []
    bytes.reserveCapacity(256 * 3)
    for (r, g, b) in customPaletteRGB {
        bytes.append(r)
        bytes.append(g)
        bytes.append(b)
    }
    return bytes
}()

let rgbLUT: [UInt8] = {
    var lut = [UInt8](repeating: 0, count: 64 * 64 * 64)
    for ir in 0..<64 {
        let r = Int(round(Double(ir) * 255.0 / 63.0))
        for ig in 0..<64 {
            let g = Int(round(Double(ig) * 255.0 / 63.0))
            for ib in 0..<64 {
                let b = Int(round(Double(ib) * 255.0 / 63.0))
                var bestIdx: UInt8 = 0
                var bestDist = Int.max
                for (pIdx, p) in customPaletteRGB.enumerated() {
                    let dr = r - Int(p.0)
                    let dg = g - Int(p.1)
                    let db = b - Int(p.2)
                    let d = dr * dr * 2 + dg * dg * 4 + db * db
                    if d < bestDist {
                        bestDist = d
                        bestIdx = UInt8(pIdx)
                        if d == 0 { break }
                    }
                }
                lut[(ir << 12) | (ig << 6) | ib] = bestIdx
            }
        }
    }
    return lut
}()

func quantizeToIndexedCGImage(_ src: CGImage) -> CGImage {
    let w = src.width
    let h = src.height
    guard let dataProvider = src.dataProvider,
          let cfData = dataProvider.data,
          let srcPtr = CFDataGetBytePtr(cfData) else {
        return src
    }
    let bpr = src.bytesPerRow
    var indexedPixels = [UInt8](repeating: 0, count: w * h)
    indexedPixels.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        rgbLUT.withUnsafeBufferPointer { lutBuf in
            guard let lutBase = lutBuf.baseAddress else { return }
            for y in 0..<h {
                let rowPtr = srcPtr + y * bpr
                let outRow = dstBase + y * w
                let rowDither = ((y & 1) == 0) ? 0 : 1
                for x in 0..<w {
                    let px = rowPtr + (x << 2)
                    var r = Int(px[0])
                    var g = Int(px[1])
                    var b = Int(px[2])
                    let m = max(r, max(g, b))
                    if m > 0 && m < 36 {
                        r = min(255, r + rowDither)
                        g = min(255, g + rowDither)
                        b = min(255, b + rowDither)
                    }
                    outRow[x] = lutBase[((r >> 2) << 12) | ((g >> 2) << 6) | (b >> 2)]
                }
            }
        }
    }

    let baseSpace = CGColorSpaceCreateDeviceRGB()
    guard let indexedSpace = paletteFlatBytes.withUnsafeBufferPointer({ buf -> CGColorSpace? in
        guard let base = buf.baseAddress else { return nil }
        return CGColorSpace(indexedBaseSpace: baseSpace, last: 255, colorTable: base)
    }) else {
        return src
    }

    let cfOut = CFDataCreate(nil, indexedPixels, indexedPixels.count)!
    guard let provider = CGDataProvider(data: cfOut),
          let outImg = CGImage(
              width: w,
              height: h,
              bitsPerComponent: 8,
              bitsPerPixel: 8,
              bytesPerRow: w,
              space: indexedSpace,
              bitmapInfo: CGBitmapInfo(rawValue: 0),
              provider: provider,
              decode: nil,
              shouldInterpolate: false,
              intent: .defaultIntent
          ) else {
        return src
    }
    return outImg
}

func savePNG(_ image: CGImage, to url: URL) {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
}

func saveGIF(frames: [CGImage], fps: Int, to url: URL) {
    let fileProps: [CFString: Any] = [
        kCGImagePropertyGIFDictionary: [
            kCGImagePropertyGIFLoopCount: 0,
            kCGImagePropertyGIFHasGlobalColorMap: true
        ]
    ]
    let frameDelay = 1.0 / Double(fps)
    let frameProps: [CFString: Any] = [
        kCGImagePropertyGIFDictionary: [
            kCGImagePropertyGIFDelayTime: frameDelay,
            kCGImagePropertyGIFUnclampedDelayTime: frameDelay
        ]
    ]
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.gif.identifier as CFString, frames.count, nil) else { return }
    CGImageDestinationSetProperties(dest, fileProps as CFDictionary)
    for f in frames {
        let indexed = quantizeToIndexedCGImage(f)
        CGImageDestinationAddImage(dest, indexed, frameProps as CFDictionary)
    }
    CGImageDestinationFinalize(dest)
}

func saveMP4(frames: [CGImage], fps: Int, to url: URL) {
    try? FileManager.default.removeItem(at: url)
    guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return }
    let settings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: Int(W),
        AVVideoHeightKey: Int(H),
        AVVideoCompressionPropertiesKey: [
            AVVideoAverageBitRateKey: 16_000_000
        ]
    ]
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32ARGB),
            kCVPixelBufferWidthKey as String: Int(W),
            kCVPixelBufferHeightKey as String: Int(H)
        ]
    )
    writer.add(input)
    writer.startWriting()
    writer.startSession(atSourceTime: .zero)

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    for (idx, img) in frames.enumerated() {
        while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.005) }
        var pb: CVPixelBuffer?
        guard let pool = adaptor.pixelBufferPool,
              CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb) == kCVReturnSuccess,
              let pixelBuffer = pb else { continue }
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        if let pxCtx = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: Int(W),
            height: Int(H),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) {
            pxCtx.draw(img, in: CGRect(x: 0, y: 0, width: W, height: H))
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        let time = CMTime(value: Int64(idx), timescale: Int32(fps))
        adaptor.append(pixelBuffer, withPresentationTime: time)
    }
    input.markAsFinished()
    let sem = DispatchSemaphore(value: 0)
    writer.finishWriting { sem.signal() }
    sem.wait()
}

var outDir = "/Users/jonathan/Desktop/nlp-project/docs/slides"
var slidesToRender: [Int] = Array(1...15)
var pngOnly = false

let args = CommandLine.arguments
var i = 1
while i < args.count {
    if (args[i] == "--output-dir" || args[i] == "--slides-dir"), i + 1 < args.count {
        outDir = args[i + 1]
        i += 2
    } else if (args[i] == "--slides" || args[i] == "--only"), i + 1 < args.count {
        slidesToRender = args[i + 1].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        i += 2
    } else if args[i] == "--png-only" {
        pngOnly = true
        i += 1
    } else {
        i += 1
    }
}

let fm = FileManager.default
try? fm.createDirectory(atPath: "\(outDir)/png", withIntermediateDirectories: true)
try? fm.createDirectory(atPath: "\(outDir)/gifs", withIntermediateDirectories: true)
try? fm.createDirectory(atPath: "\(outDir)/videos", withIntermediateDirectories: true)

let fps = 8

for s in slidesToRender {
    let name = String(format: "slide_%02d", s)
    let totalFrames = (s == 1 || s == 2 || s == 15) ? 120 : 72

    let hiResPNG = renderFrame(slideIndex: s, t: 0.0, scale: 2.0)
    savePNG(hiResPNG, to: URL(fileURLWithPath: "\(outDir)/png/\(name).png"))
    if pngOnly {
        print("Rendered \(name) (4K .png)")
        continue
    }

    var frames: [CGImage] = []
    frames.reserveCapacity(totalFrames)
    for f in 0..<totalFrames {
        let t = CGFloat(f) / CGFloat(totalFrames)
        frames.append(renderFrame(slideIndex: s, t: t, scale: 1.0))
    }
    saveGIF(frames: frames, fps: fps, to: URL(fileURLWithPath: "\(outDir)/gifs/\(name).gif"))
    saveMP4(frames: frames, fps: fps, to: URL(fileURLWithPath: "\(outDir)/videos/\(name).mp4"))
    print("Rendered \(name) (4K .png, \(totalFrames)f HD .gif, HD .mp4)")
}
