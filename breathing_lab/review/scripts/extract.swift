import Foundation
import AVFoundation
import AppKit
import Accelerate
let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().path
let asset = AVURLAsset(url: URL(fileURLWithPath: root + "/../data/raw/ml-breath.mov"))
let duration = CMTimeGetSeconds(try await asset.load(.duration))
let tracks = try await asset.loadTracks(withMediaType: .video)
let track = tracks[0]
let size = try await track.load(.naturalSize)
let fps = try await track.load(.nominalFrameRate)
let formats = try await track.load(.formatDescriptions)
let info: [String:Any] = ["duration":duration,"width":size.width,"height":size.height,"nominal_fps":fps,"codec":String(format: "%c%c%c%c", (CMFormatDescriptionGetMediaSubType(formats[0]) >> 24)&255,(CMFormatDescriptionGetMediaSubType(formats[0]) >> 16)&255,(CMFormatDescriptionGetMediaSubType(formats[0]) >> 8)&255,CMFormatDescriptionGetMediaSubType(formats[0])&255)]
try JSONSerialization.data(withJSONObject:info,options:.prettyPrinted).write(to:URL(fileURLWithPath:root+"/metadata.json"))
print(info)
let generator = AVAssetImageGenerator(asset:asset)
generator.appliesPreferredTrackTransform = true
generator.maximumSize = NSSize(width:960,height:960)
generator.requestedTimeToleranceBefore = .zero
generator.requestedTimeToleranceAfter = .zero
for (i, seconds) in [2.0,duration*0.2,duration*0.4,duration*0.6,duration*0.8,duration-2].enumerated() {
 let (cg,actual) = try await generator.image(at:CMTime(seconds:seconds,preferredTimescale:600))
 try NSBitmapImageRep(cgImage:cg).representation(using:.png,properties:[:])!.write(to:URL(fileURLWithPath:root+"/sample-\(i).png"))
 print("sample \(i): \(CMTimeGetSeconds(actual))")
}
let reader = try AVAssetReader(asset:asset)
let output = AVAssetReaderTrackOutput(track:track,outputSettings:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA])
output.alwaysCopiesSampleData=false
reader.add(output)
guard reader.startReading() else {fatalError("reader failed")}
FileManager.default.createFile(atPath:root+"/green.bin",contents:nil)
let file = try FileHandle(forWritingTo:URL(fileURLWithPath:root+"/green.bin"))
let w=320,h=180
var small=[UInt8](repeating:0,count:w*h*4)
var green=[UInt8](repeating:0,count:w*h)
var times=[Double]()
var nextTime=0.0
while let sample = output.copyNextSampleBuffer() {
 let seconds=CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
 if seconds+0.001 < nextTime {continue}
 guard let pixel=CMSampleBufferGetImageBuffer(sample) else {continue}
 CVPixelBufferLockBaseAddress(pixel,.readOnly)
 var src=vImage_Buffer(data:CVPixelBufferGetBaseAddress(pixel)!,height:vImagePixelCount(CVPixelBufferGetHeight(pixel)),width:vImagePixelCount(CVPixelBufferGetWidth(pixel)),rowBytes:CVPixelBufferGetBytesPerRow(pixel))
 small.withUnsafeMutableBytes { bytes in
  var dst=vImage_Buffer(data:bytes.baseAddress!,height:vImagePixelCount(h),width:vImagePixelCount(w),rowBytes:w*4)
  vImageScale_ARGB8888(&src,&dst,nil,vImage_Flags(kvImageNoFlags))
 }
 CVPixelBufferUnlockBaseAddress(pixel,.readOnly)
 for j in 0..<(w*h) {green[j]=small[j*4+1]}
 try file.write(contentsOf:Data(green))
 times.append(seconds)
 nextTime=seconds+0.095
}
try file.close()
try JSONSerialization.data(withJSONObject:["times":times,"width":w,"height":h,"reader_status":reader.status.rawValue],options:[]).write(to:URL(fileURLWithPath:root+"/frames.json"))
print("extracted \(times.count) frames, status \(reader.status.rawValue), error \(String(describing:reader.error))")
