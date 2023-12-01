# OME-Zarr tools for the Vesuvius scrolls

### What is OME-Zarr?

Zarr is a file format that facilitates efficient access
to huge data sets stored in zarr data volumes.

The zarr format also specifies how to store hierarchical sets
("groups") of
zarr data volumes, and it supports metadata files.

OME-Zarr is a layer on top of zarr that
allows multi-resolution data volumes to be stored and accessed in 
a standardized way.
OME-Zarr does this by 
specifying a convention for structuring the zarr groups and
metadata that hold these data volumes.

The zarr-python
library provides an API that allows the programmer to easily
retrieve zarr and OME-Zarr
data, as if it were stored in a numpy array, thus hiding the
complexity of the actual storage mechanism.

### OME-Zarr and Neuroglancer

Neuroglancer is a program, developed at Google (though
not an official Google product),
that allows the efficient exploration of data volumes stored in
various formats, including the OME-Zarr format.
By retrieving only the necessary data,
at the necessary resolution, Neuroglancer lets the user move
easily and efficiently through enormous data volumes.

### OME-Zarr and khartes

Brett Olsen (active on the scrollprize.org Discord server)
has recently added code that allows khartes to access TIFF files
as if they were zarr data stores.
This includes
the 500x500x500 grid files, created by @spelufo (same Discord server)
that are available in each scroll's `volume_grid` directory
on the scrollprize.org data server.
As a result, users can now
navigate through an entire scroll volume, instead
of having to create a limited-size khartes data volume
in advance.
However, the data
in this case is not multi-resolution, so navigating the data
is not as efficient as with Neuroglancer.

A next step will be to allow khartes
to access zarr data stores directly, and eventually to
take advantage of OME-Zarr multi-resolution data stores.

### What is in this repository?

In order for khartes to eventually work with multi-resolution
scroll data in
OME-Zarr format, there needs to be a way to convert the 
existing scroll
data to this format.  So the main script in this repository
is `scroll_to_ome`, which performs this conversion.

Another script, `ppm_to_layers`, reads a `.ppm` file and
zarr-format scroll data, and uses these to create a flattened
layer volume that is as similar as possible to
the one created by `vc_layers_from_ppm`.  This was written to
test how well the zarr format works in practice, and to learn
how best to take advantage of the zarr format's capabilities.

### The rest of this README

Here is what you will find in the rest of this README file.

* A guide to installing the scripts. 
* Detailed instructions on how to use `scroll_to_ome` to create
an OME-Zarr data store from a scroll's TIFF files.
* A further discussion of OME-Zarr.
* Not-very-detailed instructions on how to run `ppm_to_layers`
* A beginner's guide to setting up Neuroglancer to view the 
OME-Zarr files created by `scroll_to_ome`

## Installation

Only a few packages need to be installed in order to run
`scroll_to_ome`:
* zarr
* tifffile
* scikit-image

When these packages are installed, other required packages, such
as numpy, will automatically be brought in, so they are not 
explicitly listed here.
The file `anaconda_installation.txt` lists the conda commands
that will import these packages, if you are using anaconda.

## User guide for `scroll_to_ome`

`scroll_to_ome` converts a scroll (more precisely, the
z-slice TIFF files that store the scroll data) into a 
multi-resolution OME-Zarr data store.
`scroll_to_ome` takes a number of arguments, but only two
are mandatory: the name of the directory that contains 
the scroll TIFF files,
and the name of the directory where the OME-Zarr multi-resolution
data will be stored.

For example:
```
python scroll_to_ome.py /mnt/vesuvius/Scroll1.volpkg/volumes/20230205180739 /mnt/vesuvius/scroll1.zarr
```

There are a number of options that modify the
default behavior, but before I list
them, I need to give you a little
more background on zarr and OME-Zarr.

In the zarr data format, the data volume is
decomposed into 'chunks', which are 3D rectangles of data.
Zarr provides many ways to store these chunks,
but for the case we are interested in,
each chunk is stored in a separate file;
these files provide rapid access to data anywhere in the volume.
All chunks are the same size.
A typical chunk size is 128x128x128
voxels, but chunks can be non-cubical; 137x67x12, for
instance, is a valid 
(though not necessarily recommended) chunk size.

OME-Zarr provides a way to store multi-resolution data: that is,
multiple copies of the same data volume, 
sampled at different resolutions.
`scroll-to-ome` by default creates volumes at 6 different
resolutions (including the original full-resolution volume); 
each volume has half the resolution of the previous one.
Thus, with the default settings, the 6th volume has 1/32 the resolution
of the original.  That represents a reduction of 32x32x32 = 32,768 
in the size of the data volume, reducing a 1 Tb data volume
to 33 Mb.

One interesting feature of zarr is that if a chunk contains
only zeros, the file containing the chunk is not created at all.
This is not very helpful in the case of the original scroll files,
but @james darby (on the Discord server) has created a 
"masked" version of Scroll 1, where all the
pixels outside of the actual scroll are set to zero.  This saves
a lot of space in the OME-Zarr data store.

If you type `python scroll_to_ome.py --help` you will see all of the
options that are available.  The options default to reasonable values,
so if you leave them all alone, you will get a good result.

But in case you like tweaking things,
what follows is a description of each option.

* **chunk_size**: The size of the chunks that will be created.
`scroll-to-ome` only creates cubical chunks, so if you choose,
say, a chunk-size of 128, then the chunks will be 128x128x128
in size.
All resolutions in the multi-resolution data set will be
created with the same chunk size.

    In theory, smaller chunks should load more quickly, because
    there is less data to transfer, but there may be a fixed overhead
    cost per chunk as well.  I've chosen 128 as the default, but
    64 and 256 are also worth considering.  

* **nlevels**: "Level" refers to how many levels of resolution
will be created in the multi-resolution OME-Zarr data store.
The default, 6, means that in addition to the full-resolution
zarr store, 5 more will be created at successively lower
resolutions.

    `scroll-to-ome` always reduces the resolution by a factor
    of 2 from one level to the next, along all 3 axes, so each level
    is approximately 8 times smaller in data volume than the previous
    level (approximately, because the chunks may not line up exactly
    with the data volume boundaries).

* **max_gb**: By default, `scroll-to-ome` uses as much memory
as it needs to run efficiently; this is approximately chunk size times
TIFF size.  So for instance, if each input TIFF file is 185 Mb, and
the chunk size is 128, the program will need about 24 Gb of RAM
to run efficiently.

    If your computer is not able to provide this much memory, or
    page faults slow your computer down, you can specify the 
    maximum memory size (in Gb) that should be used.  Setting this
    means that the zarr files will need to be written repeatedly, which
    will slow down the conversion process significantly.

* **zarr_only**: This is a flag indicating that only a single
full-resolution zarr data store should be created, rather than
an OME-Zarr multi-resolution set of zarr stores.

* **overwrite**: By default, `scroll-to-ome` will not overwrite
an existing zarr data store.  Setting this flag allows overwriting;
use it with care!

* **algorithm**: There exist a number of algoithms for down-sampling
the data from one resolution level to the next.  By default,
`scroll-to-ome` takes the mean of the 8 higher-resolution voxels that 
will be combined into 1 lower-resolution voxel.  

    A second
    option is 'nearest', which simply selects the value
    of one of the high-resolution voxels and assigns it to the 
    lower-resolution one.  
    This tends to create rather jittery looking
    images.

    The third option is 'gaussian', which uses a Gaussian weighing
    function to smooth the data before sub-sampling.  This seems
    to create an overly smooth sub-sampled image, but perhaps this
    would be improved by choosing a different sigma factor.  In any
    case, the calculations significantly slow down the conversion
    process.

* **ranges**: This option is useful if you only want
to extract a certain range of data from the TIFF files.
One reason you might want to do this is to test your parameters
before committing yourself to a long conversion run.
The format along each axis is similar to the python "slice" format: 
`min:max`, where the extracted data covers the range min, min+1, min+2, ...,
max-3, max-2, max-1.  Recall that the x and y axes correspond to
the x and y axes in a single TIFF file, and z corresponds to the
number of the TIFF file.

    As with the slice format, you can omit part of the range,
    so that `:5` is the same as `0:5`, `5:` is the same as 5 to the end,
    and `:` is the same as the entire range.  This last example is
    useful when you only want to limit some of the ranges, for instance
    `:,:,:1000` will use the full xy extent of the first 1000 TIFF files.

    The slice format permits a third parameter, the step size,
    but `slice-to-ome` only allows a step size of 1.

* **first_new_level**: This option allows you to go back to an
existing OME-Zarr data store and add new levels of resolution.
To see which resolution
levels already exist, `cd` into the data store and look at the
existing directories there.  You should see one directory
per level, starting with `0`, which is the full-resolution
zarr data store, then `1`, `2`, etc.  As the name of the option 
suggests, pass it the number of the first level that is not
already there.
You may also need to set **nlevels** to a different value.

## OME-Zarr conversion: Time and space

In general, the multi-resolution data store created by
`scroll-to-ome` takes up about 20% more space than the original
TIFF files.  That is, if the scroll TIFF files occupy 1 Tb,
then the OME-Zarr data store will occupy about 1.2 Tb.

The reason: The full-resolution zarr data store will take
up about as much space as the TIFF files, since it is essentially
a rearrangement of the existing scroll data (but there is an
exception discussed below!).  Then each successive zarr data store
in the multi-resolution series takes up about 1/8 of the space of the
previous one, since the resolution reduction is a factor of 2
along each axis.  So the total storage space required, compared
to just the space required by the full-resolution data store,
is 1 + 1/8 + 1/64 + ..., which is somewhat less than 1.2.

It is important to know that in some cases, the zarr data store
can take up much less space than the original TIFF files!
This is thanks to one of the data compression methods used by
zarr.

Zarr offers a number of data compression methods; most of them
do not perform well on scroll data.  That is why `scroll_to_ome`
does not compress the data that is stored in the zarr data volumes.
However, one method that works well,
on certain data sets, is that the zarr library
can be set not to create chunks
that are full of zeros.  That is, when writing chunks, if the zarr
library
detects that the chunk consists of all zeros, then that chunk
is simply not written to disk.
Later, when reading the data store, if the zarr library detects that
that chunk is missing, it assumes that the chunk was all zeros.

The original scroll data does not contain many zeros, so this
simple compression scheme might seem irrelevant.  However,
@james darby has created modified TIFFs for some scrolls, available
in the scroll's `masked_volumes` directory, where pixels outside of the
scroll itself have been set to zero.  These modified TIFFs are
smaller than the original TIFFs, since a compression scheme
has been used to hide the zeros.  So a zarr data store created from
the masked_volumes TIFFs will not be any smaller than the
masked_volumes directory itself.  However, it will be significantly
smaller than a zarr data store created from the original non-masked
TIFFs, with no loss of information inside the scroll itself.

Another note: the `max_gb`
flag may be useful on machines with limited RAM,
but you should test on a limited number of slices before processing
the entire scroll volume.  Depending on the speed of your swap
disk versus your data disk, it may be faster to use swap space
than to use the `max_gb` flag to limit memory usage.

As for time requirements, I only have one data point.  Using
a chunk size of 128, on a laptop with
32 Gb RAM (so the `max_gb` flag did not need to be used),
with the TIFFs and the OME data store on an external USB 3 SSD,
the conversion took less than 24 hours, but more than 12.

## OME-Zarr naming conventions

The zarr format allows multiple data volumes to be
nested hierarchically in a single zarr directory.
In fact, as mentioned before,
this is how the OME-Zarr format works: it is
simply a convention for how the multi-resolution data volumes
should be arranged in a hierarchy within a single zarr directory.

This means that
multi-resolution OME-Zarr data stores,
as well
as single high-resolution zarr data stores,
are located in directories that by convention end with a `.zarr`
suffix.
One way to see whether a `.zarr` directory contains a single
data volume, or a set of multi-resolution data volumes, is
to go into the directory.

If you see a file named `.zarray`,
then that directory contains a single data volume.

If you see files name `.zattrs` and `.zgroup`, then the
directory probably contains OME-Zarr data.

To find the high-resolution zarr data volume in an OME-Zarr
directory, go into that directory and look for the sub-directory
named `0`.  Although the name of the `0` directory does not
end in the conventional `.zarr`, it is in fact a zarr data store,
containing the high-resolution zarr data.

## User guide for `ppm_to_layers`

Like `vc_layers_from_ppm`, `ppm_to_layers` takes a `.ppm` file
(generated by `vc_render`), and the scroll data, and from these
creates a flattened layer volume (a directory full of TIFFs,
one per z value in the flattened volume).
The main difference between
the two programs is that `vc_layers_from_ppm` uses the scroll TIFF
files, whereas `ppm_to_layers` uses the full-resolution zarr data.

`ppm_to_layers` requires the user to provide the name of a `.ppm`
file, and the name of a zarr data store containing a single data
volume.  Note that giving the name of an OME-Zarr directory won't
work.  However, as the "OME-Zarr naming convention" section above
indicates, you can simply append a `/0` (or `\0` on Windows) to
the name of the OME-Zarr directory, and then `ppm_to_layers` should
work.

Sometimes, in addition to the flattened data volume, you might
want to create
a single TIFF file representing a stack of the flattened data,
which is generated by finding the maximum value at each uv pixel
over a range of layers.  This is equivalent to the single TIFF
file created by `vc_render`.  To create such a file, simply put
the desired name of this file on the command line, following the
names of the `.ppm` file, the zarr directory, and the output
TIFF directory.

If you type `python ppm_to_layers.py --help` you will see all of the
options that are available.  The options default to reasonable values,
so if you leave them all alone, you will get a good result.

* **number_of_layers**: This is the number of flattened layers 
(TIFF files) that will be created.  65 is the default used
by `vc_layers_from_ppm` and `vc_render`.  The two VC programs
provide additional flexibility in terms of layer spacing, and whether
the layers will be symmetrical or assymetrical around the center,
but these options currently do not exist in `ppm_to_layers`.

* **number_of_stacked_layers**: This is the number of layers that
will be stacked together to form a single stacked TIFF file.
The stacked TIFF file is similar to the one created by `vc_render`.
I have not yet figured out the exact number that `vc_render` uses,
but the resulting TIFF file is fairly similar.  `vc_render` provides
a number of options on how to stack the layers together (find the
maximum, take the average, etc.), but `ppm_to_layers` provides
only one option: take the maximum value at each uv location, which
is the default used by `vc_render`.

* **block_size** is an advanced option.  The algorithm used by
`ppm_to_layers` breaks the PPM data into smaller blocks, to
conserve memory.  This number specifies the size of each block.

* **zarr_cache_max_size_gb** is an advanced option.  The zarr
data, as it is read, is stored in a cache.  Ideally, the cache
should be large enough so that there are few cache misses, but
not so large that it wastes memory.  In limited
experiments, the default size
(8 Gb) has met these criteria.

## A beginner's guide to using Neuroglancer to view OME-Zarr data

Neuroglancer (https://github.com/google/neuroglancer) is a
browser-based application for viewing 
volumetric data.
It is hosted under Google's github account, but it is not
an official Google product.

Neuroglancer is able to browse OME-Zarr data stores, taking
advantage of the multi-resolution data they contain.  

In this section, I will show you how to set up Neuroglancer
to browse an OME-Zarr data store on your computer.

The main thing to understand is that as a web-based application,
Neuroglancer must be run via a web server.  Neuroglancer also
expects the OME-Zarr data to be served by a web server.
Fortunately, these two servers are pretty easy to set up.

1. Build and start Neuroglancer.
    1. Download or clone Neuroglancer from github: 
https://github.com/google/neuroglancer
    2. Follow the instructions in the Building section
of the Neuroglancer README file.  If your experience 
is like mine, after you type `npm i`, you will at one point
see some dire security warnings flash by.  To me this suggests
that you might not want to expose your local Neuroglancer server to
the public internet.
    3. When you start the local server (`npm run dev-server`), you
should be sure that you are still in the neuroglancer directory.
    4. You should now be able to use a browser on your local machine
to access Neuroglancer on http://localhost:8080

2. Start a data server.
    1. In a different terminal window, go to the directory that you
want to serve data from.  This might be the directory that
contains your OME-Zarr `.zarr` directory.
    2. Use python to run the script called `cors_webserver.py`, which
is in the neuroglancer directory.  This is to avoid problems
with CORS (https://en.wikipedia.org/wiki/Cross-origin_resource_sharing),
which you would probably be happier not knowing anything about.
    3. By default, the data server is now making data available
on http://localhost:9000 .

3. Go to the Neuroglancer window that is in your browser.  
    1. From the list on the right, 
select `zarr2://` (Zarr v2 data source).
This will put `zarr2://` into the Source line on the top right.
    2. From the new list on the right,
select `http://`  The Source line will now show `zarr2://http://`
    3. Now you will type directly in the source line.  Just after
the current contents, type `localhost:9000/` (don't forget the
trailing slash!) so the source line should show:
`zarr2://http://localhost:9000/`
    4. At this point you should see a list of the contents of
the directory where your data server is running, and in the window
where you started the data server, you should see signs that
the server is indeed serving data.
    5. Select your OME-Zarr directory.
    6. The list may show the contents of this directory (`0/`, `1/`, etc),
but don't click on any of these!  Instead, while your cursor is
still in the Source window, hit Enter.
    7. If all goes well, Neuroglancer should now
show you some information about your data volume.  Almost there!
    8. At the bottom of the information, there should be a yellow
bar that reads: `Create as **image** layer`.  Click this.
    9. Your data should now begin showing up in the Neurglancer
window.  Happy exploring!
    10. If the data did not show up, go to the window where you
started your data server, and look for clues...

So that is how to use Neuroglancer to view your data.

One thing to keep in mind, if your data volumes are located
on a password-protected web server, is that Neuroglancer is
not programmed to work with http password protection, 
so you cannot
directly view password-protected data.  In this case, you will
need to run, on your local machine, a proxy server which
downloads data from the password-protected server
as needed.  Then point Neuroglancer to your local proxy server.
And if you figure out how to make a proxy server,
please submit a Pull Request!
