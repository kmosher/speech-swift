import AVFoundation
import Foundation
import MLX

public struct AudioUtilities {
    enum AudioError: Error {
        case unableToLoadAudioFile
        case unableToSaveAudioFile
    }
    
    public static func loadAudioFile(url: URL) throws -> MLXArray {
        let input = try AVAudioFile(forReading: url)
        let frameCount = AVAudioFrameCount(input.length)
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: input.processingFormat, frameCapacity: frameCount) else {
            throw AudioError.unableToLoadAudioFile
        }
        try input.read(into: inputBuffer)
        let targetFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: input.processingFormat.sampleRate, channels: 1, interleaved: false)!
        let converter = AVAudioConverter(from: input.processingFormat, to: targetFormat)!
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: frameCount) else {
            throw AudioError.unableToLoadAudioFile
        }
        try converter.convert(to: outputBuffer, from: inputBuffer)
        let samples = Array(UnsafeBufferPointer(start: outputBuffer.int16ChannelData![0], count: Int(outputBuffer.frameLength)))
        return MLXArray(converting: samples.map { Double($0) / Double(Int16.max) })
    }
    
    public static func saveAudioFile(url: URL, samples: MLXArray) throws {
        let samples = samples.asArray(Float.self)
        
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24000, channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw AudioError.unableToSaveAudioFile
        }
        
        guard let channelData = buffer.floatChannelData?[0] else {
            throw AudioError.unableToSaveAudioFile
        }
        
        samples.enumerated().forEach { channelData[$0.offset] = $0.element }
        buffer.frameLength = buffer.frameCapacity
        
        let outputFile = try AVAudioFile(forWriting: url, settings: format.settings)
        try outputFile.write(from: buffer)
    }
}
