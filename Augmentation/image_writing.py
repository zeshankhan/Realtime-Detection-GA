# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:44:02 2022

@author: Zeshan Khan
"""

def file_process_in_memory(images):
    """ Converts PIL image objects into BytesIO in-memory bytes buffers. """
    for i, (image_name, pil_image) in enumerate(images):
        file_object = io.BytesIO()
        pil_image.save(file_object, "JPEG")
        pil_image.close()
        images[i][1] = file_object  # Replace PIL image object with BytesIO memory buffer.
    return images  # Return modified list.
