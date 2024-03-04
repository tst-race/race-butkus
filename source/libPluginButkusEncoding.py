
# Copyright 2023 TwoSix
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

#sdk stuff
from commsPluginBindings import (
    COMPONENT_OK,
    #COMPONENT_ERROR,
    IEncodingSdk,
    ENCODE_OK,
    ENCODE_FAILED,
    IEncodingComponent,
    COMPONENT_STATE_STARTED,
    EncodingProperties,
    SpecificEncodingProperties,
)

#mbfte stuff
from PluginButkus.mbfte.textcover import TextCover

from PluginButkus.mbfte.pytorch_model_wrapper import (
    PyTorchModelWrapper,
    #TopKModelWrapper,
    #VariableTopKModelWrapper,
)

from PluginButkus.Log import (
    logDebug,
    #logError,
    #logInfo,
    logWarning,
)

import traceback

#bug workaround
import sys
if not sys.argv:
  sys.argv.append("(C++)")
#end bug workaround

class butkus(IEncodingComponent):
    def __init__(self, sdk, pluginConfig):
        IEncodingComponent.__init__(self)
        self.encoders = {}
        self.decoders = {}
        self.fname_length = 10
        self.sdk = sdk
        #PyTorchModelWrapper.NAME: PyTorchModelWrapper,
        #TopKModelWrapper.NAME: TopKModelWrapper,
        #VariableTopKModelWrapper.NAME: VariableTopKModelWrapper
        modelWrap = PyTorchModelWrapper

        #necessary to avoid incorrect length errors
        keyWrap = bytes.fromhex("c960efce6667ec5ca16851425a4619aa096ec8cb143d83eac3d4fc271be3a626")
        self.model = TextCover(
            # Hardcoded path pending composition pluginDirectory fix in SDK
            model_dir= f"{pluginConfig.pluginDirectory}/model",
            model_wrapper=modelWrap,
            model_params={},
            seed="Here is the news of the day. ",
            key=keyWrap,
            padding=0,
            precision=None,
            extra_encoding_bits=16,
            flip_distribution=False,
        )
        self.sdk.updateState(COMPONENT_STATE_STARTED)


    def getEncodingProperties(self) -> EncodingProperties:
        enc_props = EncodingProperties()
        #fails with message-to-encode byte size >= 96 bytes
        enc_props.encodingTime = 60
        #change this to a star for a wildcard
        enc_props.type = "*"
        return enc_props
    
    def getEncodingPropertiesForParameters(self, enc_param) -> SpecificEncodingProperties:
        sep = SpecificEncodingProperties()
        # TODO: May be able to set max byte size for encode here
        sep.maxBytes = 11*1024
        return sep
    
    def encodeBytes(self, handle, params, enc_bytes):
        # Return empty bytestring if input bytestring is empty
        if not enc_bytes:
            self.sdk.onBytesEncoded(handle, [], ENCODE_OK)
            return COMPONENT_OK

        #need to convert tuple of ints to a bytes object
        enc_bytes = bytes(enc_bytes)

        
        try:
            (covertBytes, _) = self.model.encode(plaintext=enc_bytes, complete_sentence=False)
            #convenience print the resulting covert text here
            logDebug(f"covertBytes is: {covertBytes}")

            #need to reverse the covertBytes type into the same type as enc_bytes on input
            covertBytesToSdk = covertBytes.encode('utf-8')

            #tell the sdk 
            self.sdk.onBytesEncoded(handle, covertBytesToSdk, ENCODE_OK)
            logDebug("encoding success")
            return COMPONENT_OK
        except Exception as exc:
            logWarning(f"Encode bytes failed: {exc}")
            logWarning(f"Input enc_bytes is: {enc_bytes}")
            logWarning(traceback.format_exc())
            self.sdk.onBytesEncoded(handle, [], ENCODE_FAILED)
            return COMPONENT_OK
    
    
    def decodeBytes(self, handle, params, dec_bytes):
        # Return empty bytestring if input bytestring is empty
        if not dec_bytes:
            self.sdk.onBytesEncoded(handle, [], ENCODE_OK)
            return COMPONENT_OK

        #need to convert tuple of ints to a bytes object
        dec_bytes = bytes(dec_bytes)

        reformattedDecBytes = dec_bytes.decode('utf-8')

        try :
            #do the sentinel verification check
            isVerified = self.model.check(reformattedDecBytes)
            if isVerified:
                plaintext = self.model.decode(reformattedDecBytes)
                #tell sdk what plaintext was successfully decoded 
                self.sdk.onBytesDecoded(handle, plaintext, ENCODE_OK)
                logDebug("decode success, resulting text is: " + plaintext)
            else:
                logDebug("sentinel verification check failed")
                self.sdk.onBytesDecoded(handle, [], ENCODE_FAILED)
            return COMPONENT_OK
        except Exception as exc:
            logWarning(f"Decode bytes failed: {exc}")
            logWarning(f"dec_bytes is: {dec_bytes}")
            logWarning(traceback.format_exc())
            self.sdk.onBytesDecoded(handle, [], ENCODE_FAILED)
            return COMPONENT_OK

    
def createEncoding(name: str, sdk: IEncodingSdk, roleName: str, pluginConfig):
    logDebug("The name is: " + name)
    if name == "butkus":
        temp = butkus(sdk, pluginConfig)
        return temp
    return None
